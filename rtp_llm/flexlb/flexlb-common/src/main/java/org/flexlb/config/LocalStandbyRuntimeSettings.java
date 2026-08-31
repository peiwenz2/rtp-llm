package org.flexlb.config;

import org.flexlb.dao.route.LocalStandbyConfig;
import org.flexlb.dao.route.ServiceRoute;

/**
 * Local Standby settings that can be applied without rebuilding cache-match components.
 */
public record LocalStandbyRuntimeSettings(
        long maximumEntries,
        double capacityMultiplier,
        long ttlMs,
        long minimumTtlMs,
        double ttlReductionStartRatio) {

    /**
     * Extracts and validates the Local Standby fields that can change at runtime.
     */
    public static LocalStandbyRuntimeSettings from(FlexlbConfig config) {
        if (config == null || config.getModelServiceConfig() == null) {
            throw new IllegalArgumentException("modelServiceConfig must not be null");
        }
        ServiceRoute serviceRoute = config.getModelServiceConfig();
        if (serviceRoute.getKvcm() == null || !serviceRoute.getKvcm().isEnabled()) {
            throw new IllegalArgumentException("modelServiceConfig must retain an enabled KVCM configuration");
        }
        return from(serviceRoute.getKvcm().getLocalStandby());
    }

    /**
     * Extracts and validates the Local Standby fields that can change at runtime.
     */
    public static LocalStandbyRuntimeSettings from(LocalStandbyConfig config) {
        LocalStandbyConfig localStandby = config == null ? new LocalStandbyConfig() : config;
        long maximumEntries = localStandby.getMaximumEntries();
        double capacityMultiplier = localStandby.getCapacityMultiplier();
        long ttlMs = localStandby.getTtlMs();
        long minimumTtlMs = localStandby.getMinimumTtlMs();
        double ttlReductionStartRatio = localStandby.getTtlReductionStartRatio();
        if (maximumEntries <= 0
                || !Double.isFinite(capacityMultiplier)
                || capacityMultiplier < 1.0
                || ttlMs <= 0
                || minimumTtlMs <= 0
                || minimumTtlMs > ttlMs
                || !Double.isFinite(ttlReductionStartRatio)
                || ttlReductionStartRatio <= 0
                || ttlReductionStartRatio >= 1) {
            throw new IllegalArgumentException("local standby runtime settings are invalid");
        }
        return new LocalStandbyRuntimeSettings(
                maximumEntries, capacityMultiplier, ttlMs, minimumTtlMs, ttlReductionStartRatio);
    }
}
