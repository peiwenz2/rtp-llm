package org.flexlb.config;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.service.config.ConfigSource;
import org.flexlb.util.JsonUtils;
import org.springframework.context.annotation.DependsOn;
import org.springframework.stereotype.Component;

import javax.annotation.PreDestroy;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;
import java.util.function.Function;

@Slf4j
@Component
@DependsOn({"environmentConfigSource", "nacosConfigSource"})
public class ConfigService {

    private static final List<ConfigSource> CONFIG_SOURCES = new ArrayList<>();

    private final AtomicReference<FlexlbConfig> currentConfig;
    private final List<ConfigUpdateListener> updateListeners = new ArrayList<>();

    /**
     * Guards configuration-source initialization, snapshot replacement and listener registration.
     * User-provided listeners are never invoked while this lock is held.
     */
    private final Object updateLock = new Object();

    /**
     * Serializes listener initialization and update delivery so each listener observes projected
     * values in configuration order. When both locks are needed, this lock is acquired first.
     */
    private final Object notificationLock = new Object();

    public ConfigService() {
        this.currentConfig = new AtomicReference<>(new FlexlbConfig());
        CONFIG_SOURCES.sort(Comparator.comparingInt(ConfigSource::priority));
        initializeConfigSources();
    }

    public static synchronized void register(ConfigSource source) {
        CONFIG_SOURCES.add(source);
    }

    public FlexlbConfig loadBalanceConfig() {
        return currentConfig.get();
    }

    /**
     * Registers a runtime setting derived from the current FlexLB configuration.
     *
     * <p>The projection is evaluated before a configuration snapshot is committed, so it can
     * validate the setting by throwing an exception. The applier receives the current value
     * immediately and only receives later values when the projected setting changes.
     *
     * @param projection non-null function that derives a runtime setting from a configuration snapshot
     * @param listener non-null consumer that applies the derived runtime setting
     */
    public <T> void addUpdateListener(Function<FlexlbConfig, T> projection, Consumer<T> listener) {
        ProjectedConfigUpdateListener<T> updateListener = new ProjectedConfigUpdateListener<>(projection, listener);

        synchronized (notificationLock) {
            T initialValue;
            synchronized (updateLock) {
                initialValue = projection.apply(currentConfig.get());
            }
            listener.accept(initialValue);
            synchronized (updateLock) {
                updateListener.initialize(initialValue);
                updateListeners.add(updateListener);
            }
        }
    }

    private void initializeConfigSources() {
        ConfigSource loadingSource = null;
        try {
            synchronized (updateLock) {
                for (ConfigSource source : CONFIG_SOURCES) {
                    loadingSource = source;
                    source.setUpdateListener(content -> receiveConfigUpdate(source, content));
                    String initialContent = source.load();
                    currentConfig.set(mergeConfig(currentConfig.get(), initialContent, source.name()));
                    loadingSource = null;
                    log.info("Loaded FlexLB configuration from {} source", source.name());
                }
            }
        } catch (Exception e) {
            closeConfigSources();
            throw new IllegalStateException("Failed to initialize FlexLB configuration from "
                            + (loadingSource == null ? "configured source" : loadingSource.name()), e);
        }
    }

    private void receiveConfigUpdate(ConfigSource source, String content) {
        synchronized (notificationLock) {
            try {
                List<Runnable> notifications;
                synchronized (updateLock) {
                    FlexlbConfig newConfig = mergeConfig(currentConfig.get(), content, source.name());
                    notifications = updateListeners.stream()
                            .map(listener -> listener.prepareUpdate(newConfig))
                            .toList();
                    currentConfig.set(newConfig);
                }
                notifyListeners(notifications, source.name());
                log.info("Applied FlexLB configuration update from {} source", source.name());
            } catch (Exception e) {
                log.error(
                        "Rejected invalid FlexLB configuration update from {} source; keeping last-known-good configuration: {}",
                        source.name(),
                        e.getMessage());
            }
        }
    }

    private void notifyListeners(List<Runnable> notifications, String sourceName) {
        for (Runnable notification : notifications) {
            try {
                notification.run();
            } catch (RuntimeException e) {
                log.error("Applied FlexLB configuration update from {} source, but a runtime listener failed", sourceName, e);
            }
        }
    }

    private FlexlbConfig mergeConfig(FlexlbConfig baseConfig, String content, String sourceName) {
        if (content == null || content.isBlank()) {
            throw new IllegalArgumentException(sourceName + " configuration must not be blank");
        }

        JsonNode parsed = JsonUtils.toTreeNode(content);
        if (!(parsed instanceof ObjectNode overrides)) {
            throw new IllegalArgumentException(sourceName + " configuration must be a JSON object");
        }
        if (overrides.isEmpty()) {
            throw new IllegalArgumentException(sourceName + " configuration must contain at least one FlexlbConfig field");
        }

        ObjectNode merged = (ObjectNode) JsonUtils.toTreeNode(baseConfig);
        mergeObjectFields(merged, overrides);
        FlexlbConfig config = JsonUtils.toObject(merged, FlexlbConfig.class);
        log.debug("Resolved FlexLB configuration from {} source", sourceName);
        return config;
    }

    private void mergeObjectFields(ObjectNode base, ObjectNode overrides) {
        overrides.fields().forEachRemaining(field -> {
            JsonNode existing = base.get(field.getKey());
            JsonNode override = field.getValue();
            if (existing instanceof ObjectNode existingObject && override instanceof ObjectNode overrideObject) {
                mergeObjectFields(existingObject, overrideObject);
                return;
            }
            base.set(field.getKey(), override);
        });
    }

    @PreDestroy
    public void close() {
        closeConfigSources();
    }

    private void closeConfigSources() {
        for (ConfigSource source : CONFIG_SOURCES) {
            closeQuietly(source);
        }
        CONFIG_SOURCES.clear();
    }

    private void closeQuietly(ConfigSource source) {
        if (source == null) {
            return;
        }
        try {
            source.close();
        } catch (Exception e) {
            log.warn("Failed to close {} configuration source", source.name(), e);
        }
    }

    private interface ConfigUpdateListener {

        Runnable prepareUpdate(FlexlbConfig config);
    }

    /**
     * Applies one projected configuration value and tracks the last value applied successfully.
     *
     * <p>Each candidate snapshot is projected before it is committed. Unchanged values return a
     * no-op notification. The returned notification updates {@code currentValue} only after the
     * consumer succeeds, so a failed consumer is retried by a later update whose value still
     * differs from the last successful value.
     *
     * <p>Instances are accessed while {@link ConfigService#notificationLock} is held.
     */
    private static class ProjectedConfigUpdateListener<T> implements ConfigUpdateListener {

        private final Function<FlexlbConfig, T> projection;
        private final Consumer<T> listener;
        private T currentValue;

        private ProjectedConfigUpdateListener(Function<FlexlbConfig, T> projection, Consumer<T> listener) {
            this.projection = projection;
            this.listener = listener;
        }

        private void initialize(T initialValue) {
            currentValue = initialValue;
        }

        @Override
        public Runnable prepareUpdate(FlexlbConfig config) {
            T updatedValue = projection.apply(config);
            if (Objects.equals(currentValue, updatedValue)) {
                return () -> {};
            }
            return () -> {
                listener.accept(updatedValue);
                currentValue = updatedValue;
            };
        }
    }

}
