package org.flexlb.service.config;

import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.LBConsistencyConfig;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import uk.org.webcompere.systemstubs.environment.EnvironmentVariables;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;
import java.util.function.Function;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class ConfigServiceTest {

    private ConfigService configService;

    @AfterEach
    void closeConfigService() {
        if (configService != null) {
            configService.close();
        }
    }

    @Test
    void loadsEnabledConfigSourcesByPriority() {
        EnvironmentConfigSource environmentSource = environmentSource(Map.of(
                "ENABLE_QUEUEING", "true",
                "MAX_RETRY_COUNT", "4"));
        FakeConfigSource nacosSource = new FakeConfigSource(
                "Nacos",
                200,
                "{\"enableQueueing\":false,\"maxRetryCount\":9}");

        ConfigService service = createService(List.of(nacosSource, environmentSource));

        assertThat(service.loadBalanceConfig().isEnableQueueing()).isFalse();
        assertThat(service.loadBalanceConfig().getMaxRetryCount()).isEqualTo(9);
    }

    @Test
    void doesNotLoadUnregisteredConfigSources() {
        FakeConfigSource unregisteredSource = new FakeConfigSource("unregistered", 200, "{}");

        ConfigService service = createService(List.of(
                environmentSource(Map.of("MAX_RETRY_COUNT", "4"))));

        assertThat(unregisteredSource.loaded).isFalse();
        assertThat(service.loadBalanceConfig().getMaxRetryCount()).isEqualTo(4);
    }

    @Test
    void failsFastWhenInitialSourceReadFails() {
        FakeConfigSource source = new FakeConfigSource(
                "Nacos",
                200,
                new IllegalStateException("Nacos unavailable"));

        assertThatThrownBy(() -> createService(List.of(
                environmentSource(Map.of()),
                source)))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("Failed to initialize FlexLB configuration from Nacos")
                .hasRootCauseMessage("Nacos unavailable");
        assertThat(source.closed).isTrue();
    }

    @Test
    void failsFastForMissingEmptyOrInvalidInitialContent() {
        assertInvalidInitialContent(null, "must not be blank");
        assertInvalidInitialContent("  ", "must not be blank");
        assertInvalidInitialContent("{}", "at least one FlexlbConfig field");
        assertInvalidInitialContent("[]", "must be a JSON object");
        assertInvalidInitialContent("{\"loadBalanceStrategy\":\"INVALID\"}", "not one of the values accepted");
    }

    @Test
    void ignoresUnknownSourceFields() {
        FakeConfigSource source = new FakeConfigSource(
                "Nacos",
                200,
                "{\"unknownField\":1,\"maxRetryCount\":9}");

        ConfigService service = createService(List.of(
                environmentSource(Map.of()),
                source));

        assertThat(service.loadBalanceConfig().getMaxRetryCount()).isEqualTo(9);
    }

    @Test
    void letsJacksonConvertSourceFieldValues() {
        FakeConfigSource source = new FakeConfigSource(
                "Nacos",
                200,
                "{\"enableQueueing\":\"true\",\"maxRetryCount\":\"9\"}");

        ConfigService service = createService(List.of(
                environmentSource(Map.of()),
                source));

        assertThat(service.loadBalanceConfig().isEnableQueueing()).isTrue();
        assertThat(service.loadBalanceConfig().getMaxRetryCount()).isEqualTo(9);
    }

    @Test
    void loadsNestedConsistencyConfigFromNacos() {
        FakeConfigSource source = new FakeConfigSource(
                "Nacos",
                200,
                "{\"flexlbSyncConsistencyConfig\":{\"needConsistency\":true,"
                        + "\"masterElectType\":\"ZOOKEEPER\",\"zookeeperConfig\":{"
                        + "\"zkHost\":\"zk:2181\",\"zkTimeoutMs\":10000}}}");

        ConfigService service = createService(List.of(
                environmentSource(Map.of()),
                source));
        LBConsistencyConfig consistencyConfig =
                service.loadBalanceConfig().getFlexlbSyncConsistencyConfig();

        assertThat(consistencyConfig.isNeedConsistency()).isTrue();
        assertThat(consistencyConfig.getMasterElectType()).isEqualTo(LBConsistencyConfig.MasterElectType.ZOOKEEPER);
        assertThat(consistencyConfig.getZookeeperConfig().getZkHost()).isEqualTo("zk:2181");
        assertThat(consistencyConfig.getZookeeperConfig().getZkTimeoutMs()).isEqualTo(10000);
    }

    @Test
    void letsNacosOverrideEnvironmentModelServiceConfig() {
        EnvironmentConfigSource environmentSource = environmentSource(Map.of(
                "MODEL_SERVICE_CONFIG",
                "{\"service_id\":\"environment-service\",\"role_endpoints\":[]}"));
        FakeConfigSource nacosSource = new FakeConfigSource(
                "Nacos",
                200,
                "{\"modelServiceConfig\":{\"service_id\":\"nacos-service\",\"role_endpoints\":[]}}");

        ConfigService service = createService(List.of(environmentSource, nacosSource));

        assertThat(service.loadBalanceConfig().getModelServiceConfig().getServiceId())
                .isEqualTo("nacos-service");
    }

    @Test
    void runtimeUpdatesKeepCurrentValuesForMissingFieldsAndReplaceSnapshot() {
        FakeConfigSource source = new FakeConfigSource(
                "Nacos",
                200,
                "{\"enableQueueing\":false,\"maxRetryCount\":9}");
        ConfigService service = createService(List.of(
                environmentSource(Map.of(
                        "ENABLE_QUEUEING", "true",
                        "MAX_RETRY_COUNT", "4")),
                source));

        FlexlbConfig initialSnapshot = service.loadBalanceConfig();
        source.emit("{\"enableQueueing\":false}");
        FlexlbConfig updatedSnapshot = service.loadBalanceConfig();

        assertThat(updatedSnapshot).isNotSameAs(initialSnapshot);
        assertThat(initialSnapshot.getMaxRetryCount()).isEqualTo(9);
        assertThat(updatedSnapshot.getMaxRetryCount()).isEqualTo(9);
        assertThat(updatedSnapshot.isEnableQueueing()).isFalse();
    }

    @Test
    void notifiesListenerWithCurrentAndRuntimeConfigurations() {
        FakeConfigSource source = new FakeConfigSource("Nacos", 200, "{\"maxRetryCount\":9}");
        ConfigService service = createService(List.of(
                environmentSource(Map.of()),
                source));
        List<FlexlbConfig> updates = new ArrayList<>();

        service.addUpdateListener(Function.identity(), updates::add);
        source.emit("{\"maxRetryCount\":10}");

        assertThat(updates).extracting(FlexlbConfig::getMaxRetryCount).containsExactly(9, 10);
    }

    @Test
    void publishesOnlyChangedValidatedRuntimeSettings() {
        FakeConfigSource source = new FakeConfigSource("Nacos", 200, "{\"maxRetryCount\":9}");
        ConfigService service = createService(List.of(
                environmentSource(Map.of()),
                source));
        List<Integer> updates = new ArrayList<>();

        service.addUpdateListener(config -> {
            int maxRetryCount = config.getMaxRetryCount();
            if (maxRetryCount < 0) {
                throw new IllegalArgumentException("maxRetryCount must not be negative");
            }
            return maxRetryCount;
        }, updates::add);
        FlexlbConfig lastKnownGood = service.loadBalanceConfig();

        source.emit("{\"enableQueueing\":true}");
        assertThat(updates).containsExactly(9);
        lastKnownGood = service.loadBalanceConfig();

        source.emit("{\"maxRetryCount\":-1}");
        assertThat(service.loadBalanceConfig()).isSameAs(lastKnownGood);
        assertThat(updates).containsExactly(9);

        source.emit("{\"maxRetryCount\":10}");
        assertThat(service.loadBalanceConfig()).isNotSameAs(lastKnownGood);
        assertThat(updates).containsExactly(9, 10);
    }

    @Test
    void continuesNotifyingOtherRuntimeSettingsWhenOneApplierFails() {
        FakeConfigSource source = new FakeConfigSource("Nacos", 200, "{\"maxRetryCount\":9}");
        ConfigService service = createService(List.of(
                environmentSource(Map.of()),
                source));
        List<Integer> delivered = new ArrayList<>();
        List<Integer> attempted = new ArrayList<>();
        AtomicBoolean failApplier = new AtomicBoolean(true);

        service.addUpdateListener(FlexlbConfig::getMaxRetryCount, value -> {
            attempted.add(value);
            if (value == 10 && failApplier.get()) {
                throw new IllegalStateException("test applier failure");
            }
        });
        service.addUpdateListener(FlexlbConfig::getMaxRetryCount, delivered::add);

        source.emit("{\"maxRetryCount\":10}");
        failApplier.set(false);
        source.emit("{\"maxRetryCount\":10}");
        source.emit("{\"maxRetryCount\":11}");

        assertThat(service.loadBalanceConfig().getMaxRetryCount()).isEqualTo(11);
        assertThat(delivered).containsExactly(9, 10, 11);
        assertThat(attempted).containsExactly(9, 10, 10, 11);
    }

    @Test
    @DisplayName("注册监听器时发生配置更新，不丢失初始回放之后的更新")
    void doesNotLoseUpdateThatRacesWithListenerRegistration() throws Exception {
        FakeConfigSource source = new FakeConfigSource("Nacos", 200, "{\"maxRetryCount\":9}");
        ConfigService service = createService(List.of(
                environmentSource(Map.of()),
                source));
        List<Integer> updates = Collections.synchronizedList(new ArrayList<>());
        CountDownLatch initialCallbackEntered = new CountDownLatch(1);
        CountDownLatch allowInitialCallback = new CountDownLatch(1);
        CountDownLatch updateAttempted = new CountDownLatch(1);
        CountDownLatch updateCompleted = new CountDownLatch(1);
        ExecutorService executor = Executors.newFixedThreadPool(2);

        try {
            Future<?> subscription = executor.submit(() -> service.addUpdateListener(FlexlbConfig::getMaxRetryCount, value -> {
                updates.add(value);
                if (value == 9) {
                    initialCallbackEntered.countDown();
                    awaitCallbackRelease(allowInitialCallback);
                }
            }));
            assertThat(initialCallbackEntered.await(1, TimeUnit.SECONDS)).isTrue();

            Future<?> update = executor.submit(() -> {
                try {
                    source.emit("{\"maxRetryCount\":10}", updateAttempted::countDown);
                } finally {
                    updateCompleted.countDown();
                }
            });
            assertThat(updateAttempted.await(1, TimeUnit.SECONDS)).isTrue();
            assertThat(updateCompleted.await(100, TimeUnit.MILLISECONDS)).isFalse();

            allowInitialCallback.countDown();
            subscription.get(1, TimeUnit.SECONDS);
            update.get(1, TimeUnit.SECONDS);

            assertThat(updates).containsExactly(9, 10);
        } finally {
            allowInitialCallback.countDown();
            shutdown(executor);
        }
    }

    @Test
    @DisplayName("配置快照先提交，运行时监听器按更新顺序串行执行")
    void serializesRuntimeCallbacksAfterCommittingTheSnapshot() throws Exception {
        FakeConfigSource source = new FakeConfigSource("Nacos", 200, "{\"maxRetryCount\":9}");
        ConfigService service = createService(List.of(
                environmentSource(Map.of()),
                source));
        List<Integer> updates = Collections.synchronizedList(new ArrayList<>());
        CountDownLatch firstRuntimeCallbackEntered = new CountDownLatch(1);
        CountDownLatch allowFirstRuntimeCallback = new CountDownLatch(1);
        CountDownLatch secondUpdateAttempted = new CountDownLatch(1);
        CountDownLatch secondRuntimeCallbackEntered = new CountDownLatch(1);
        AtomicInteger activeCallbacks = new AtomicInteger();
        AtomicInteger maximumConcurrentCallbacks = new AtomicInteger();
        service.addUpdateListener(FlexlbConfig::getMaxRetryCount, value -> {
            int concurrentCallbacks = activeCallbacks.incrementAndGet();
            maximumConcurrentCallbacks.updateAndGet(current -> Math.max(current, concurrentCallbacks));
            try {
                updates.add(value);
                if (value == 10) {
                    firstRuntimeCallbackEntered.countDown();
                    awaitCallbackRelease(allowFirstRuntimeCallback);
                } else if (value == 11) {
                    secondRuntimeCallbackEntered.countDown();
                }
            } finally {
                activeCallbacks.decrementAndGet();
            }
        });
        ExecutorService executor = Executors.newFixedThreadPool(2);

        try {
            Future<?> firstUpdate = executor.submit(() -> source.emit("{\"maxRetryCount\":10}"));
            assertThat(firstRuntimeCallbackEntered.await(1, TimeUnit.SECONDS)).isTrue();
            assertThat(service.loadBalanceConfig().getMaxRetryCount()).isEqualTo(10);

            Future<?> secondUpdate = executor.submit(() ->
                    source.emit("{\"maxRetryCount\":11}", secondUpdateAttempted::countDown));
            assertThat(secondUpdateAttempted.await(1, TimeUnit.SECONDS)).isTrue();
            assertThat(secondRuntimeCallbackEntered.await(100, TimeUnit.MILLISECONDS)).isFalse();

            allowFirstRuntimeCallback.countDown();
            firstUpdate.get(1, TimeUnit.SECONDS);
            secondUpdate.get(1, TimeUnit.SECONDS);

            assertThat(updates).containsExactly(9, 10, 11);
            assertThat(maximumConcurrentCallbacks).hasValue(1);
        } finally {
            allowFirstRuntimeCallback.countDown();
            shutdown(executor);
        }
    }

    @Test
    void rejectsInvalidRuntimeUpdatesAndKeepsLastKnownGoodSnapshot() {
        FakeConfigSource source = new FakeConfigSource("Nacos", 200, "{\"maxRetryCount\":9}");
        ConfigService service = createService(List.of(
                environmentSource(Map.of()),
                source));
        FlexlbConfig lastKnownGood = service.loadBalanceConfig();

        source.emit("");
        assertThat(service.loadBalanceConfig()).isSameAs(lastKnownGood);
        source.emit("{}");
        assertThat(service.loadBalanceConfig()).isSameAs(lastKnownGood);
        source.emit("{\"maxRetryCount\":\"invalid\"}");
        assertThat(service.loadBalanceConfig()).isSameAs(lastKnownGood);
    }

    @Test
    void closesRegisteredConfigSources() {
        FakeConfigSource source = new FakeConfigSource("Nacos", 200, "{\"maxRetryCount\":9}");
        ConfigService service = createService(List.of(
                environmentSource(Map.of()),
                source));

        service.close();

        assertThat(source.closed).isTrue();
    }

    private ConfigService createService(List<ConfigSource> sources) {
        for (ConfigSource source : sources) {
            if (!(source instanceof EnvironmentConfigSource)) {
                ConfigService.register(source);
            }
        }
        configService = new ConfigService();
        return configService;
    }

    private EnvironmentConfigSource environmentSource(Map<String, String> environment) {
        try {
            return new EnvironmentVariables(environment).execute(() -> {
                EnvironmentConfigSource source = new EnvironmentConfigSource();
                source.initialize();
                return source;
            });
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
    }

    private void assertInvalidInitialContent(String content, String expectedMessage) {
        FakeConfigSource source = new FakeConfigSource("Nacos", 200, content);

        assertThatThrownBy(() -> createService(List.of(
                environmentSource(Map.of()),
                source)))
                .isInstanceOf(IllegalStateException.class)
                .hasStackTraceContaining(expectedMessage);
        assertThat(source.closed).isTrue();
    }

    private void awaitCallbackRelease(CountDownLatch latch) {
        try {
            if (!latch.await(1, TimeUnit.SECONDS)) {
                throw new AssertionError("Timed out waiting for test callback release");
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new AssertionError("Interrupted while waiting for test callback release", e);
        }
    }

    private void shutdown(ExecutorService executor) throws InterruptedException {
        executor.shutdownNow();
        assertThat(executor.awaitTermination(1, TimeUnit.SECONDS)).isTrue();
    }

    private static final class FakeConfigSource implements ConfigSource {
        private final String name;
        private final int priority;
        private final String initialContent;
        private final Exception loadException;
        private Consumer<String> listener;
        private boolean loaded;
        private boolean closed;

        private FakeConfigSource(String name, int priority, String initialContent) {
            this(name, priority, initialContent, null);
        }

        private FakeConfigSource(String name, int priority, Exception loadException) {
            this(name, priority, null, loadException);
        }

        private FakeConfigSource(String name, int priority, String initialContent, Exception loadException) {
            this.name = name;
            this.priority = priority;
            this.initialContent = initialContent;
            this.loadException = loadException;
        }

        @Override
        public String name() {
            return name;
        }

        @Override
        public int priority() {
            return priority;
        }

        @Override
        public void setUpdateListener(Consumer<String> listener) {
            this.listener = listener;
        }

        @Override
        public String load() throws Exception {
            loaded = true;
            if (loadException != null) {
                throw loadException;
            }
            return initialContent;
        }

        @Override
        public void close() {
            closed = true;
        }

        private void emit(String content) {
            listener.accept(content);
        }

        private void emit(String content, Runnable beforeEmit) {
            beforeEmit.run();
            emit(content);
        }
    }
}
