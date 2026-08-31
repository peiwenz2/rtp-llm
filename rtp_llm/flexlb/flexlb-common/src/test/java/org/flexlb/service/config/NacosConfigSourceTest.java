package org.flexlb.service.config;

import com.alibaba.nacos.api.config.listener.Listener;
import org.flexlb.config.ConfigService;
import org.flexlb.config.DeploymentIdentity;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.LocalStandbyRuntimeSettings;
import org.flexlb.dao.nacos.NacosConfig;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;
import org.springframework.test.util.ReflectionTestUtils;
import uk.org.webcompere.systemstubs.environment.EnvironmentVariables;

import java.util.concurrent.atomic.AtomicReference;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.flexlb.constant.DeploymentIdentityConstants.HIPPO_ROLE;
import static org.flexlb.constant.DeploymentIdentityConstants.SPECTRUM_APPLICATION_NAME;
import static org.flexlb.constant.DeploymentIdentityConstants.SPECTRUM_DEPLOYMENT_NAME;
import static org.flexlb.constant.DeploymentIdentityConstants.SPECTRUM_WORKSPACE_ID;
import static org.flexlb.constant.NacosConfigConstants.NACOS_DATA_ID;
import static org.flexlb.constant.NacosConfigConstants.NACOS_GROUP;
import static org.flexlb.constant.NacosConfigConstants.NACOS_NAMESPACE;
import static org.flexlb.constant.NacosConfigConstants.NACOS_SERVER_ADDR;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class NacosConfigSourceTest {

    @Test
    void isDisabledWhenNacosAddressIsNotConfigured() throws Exception {
        NacosConfigSource source = new EnvironmentVariables(HIPPO_ROLE, "flexlb-test")
                .remove(NACOS_SERVER_ADDR)
                .execute(() -> new NacosConfigSource(new DeploymentIdentity()));

        source.initialize();
        ConfigService configService = new ConfigService();

        assertThat(source.priority()).isEqualTo(2);
        assertThat(configService.loadBalanceConfig().getMaxRetryCount()).isZero();
        configService.close();
    }

    @Test
    void failsFastWhenDataIdCannotBeResolved() {
        EnvironmentVariables environment = new EnvironmentVariables(NACOS_SERVER_ADDR, "127.0.0.1:8848")
                .remove(NACOS_DATA_ID)
                .remove(HIPPO_ROLE)
                .remove(SPECTRUM_WORKSPACE_ID)
                .remove(SPECTRUM_APPLICATION_NAME)
                .remove(SPECTRUM_DEPLOYMENT_NAME);

        assertThatThrownBy(() -> environment.execute(() -> new NacosConfigSource(new DeploymentIdentity())))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining(HIPPO_ROLE);
    }

    @Test
    void usesHippoRoleWhenDataIdIsNotConfigured() throws Exception {
        NacosConfigSource source = new EnvironmentVariables(
                NACOS_SERVER_ADDR, "127.0.0.1:8848",
                HIPPO_ROLE, "flexlb-hongyi-test-v1-flexlb-standalone")
                .remove(NACOS_DATA_ID)
                .remove(SPECTRUM_WORKSPACE_ID)
                .remove(SPECTRUM_APPLICATION_NAME)
                .remove(SPECTRUM_DEPLOYMENT_NAME)
                .execute(() -> new NacosConfigSource(new DeploymentIdentity()));

        assertThat(source)
                .extracting("config")
                .isEqualTo(new NacosConfig(
                        "127.0.0.1:8848",
                        "flexlb-hongyi-test-v1-flexlb-standalone",
                        null,
                        null));
    }

    @Test
    void usesSpectrumIdentityWhenDataIdIsNotConfigured() throws Exception {
        NacosConfigSource source = new EnvironmentVariables(
                NACOS_SERVER_ADDR, "127.0.0.1:8848",
                SPECTRUM_WORKSPACE_ID, "df4a7748",
                SPECTRUM_APPLICATION_NAME, "flexlb-test",
                SPECTRUM_DEPLOYMENT_NAME, "flexlb-test-wlcb",
                HIPPO_ROLE, "legacy-role")
                .remove(NACOS_DATA_ID)
                .execute(() -> new NacosConfigSource(new DeploymentIdentity()));

        assertThat(source)
                .extracting("config")
                .isEqualTo(new NacosConfig(
                        "127.0.0.1:8848",
                        "spectrum:df4a7748:flexlb-test:flexlb-test-wlcb",
                        null,
                        null));
    }

    @Test
    void loadsListensAndClosesNacosConfig() throws Exception {
        com.alibaba.nacos.api.config.ConfigService client =
                mock(com.alibaba.nacos.api.config.ConfigService.class);
        ArgumentCaptor<Listener> listenerCaptor = ArgumentCaptor.forClass(Listener.class);
        when(client.getConfig(
                org.mockito.ArgumentMatchers.eq("flexlb-test"),
                org.mockito.ArgumentMatchers.eq("FLEXLB_GROUP"),
                org.mockito.ArgumentMatchers.eq(3000L)))
                .thenReturn("{\"maxRetryCount\":9}");
        NacosConfigSource source = createSource(client, "test-namespace");

        source.initialize();
        verify(client).addListener(
                org.mockito.ArgumentMatchers.eq("flexlb-test"),
                org.mockito.ArgumentMatchers.eq("FLEXLB_GROUP"),
                listenerCaptor.capture());
        ConfigService configService = new ConfigService();

        assertThat(configService.loadBalanceConfig().getMaxRetryCount()).isEqualTo(9);
        listenerCaptor.getValue().receiveConfigInfo("{\"maxRetryCount\":10}");
        configService.close();

        assertThat(configService.loadBalanceConfig().getMaxRetryCount()).isEqualTo(10);
        verify(client).removeListener(
                "flexlb-test",
                "FLEXLB_GROUP",
                listenerCaptor.getValue());
        verify(client).shutDown();
    }

    @Test
    void appliesNestedLocalStandbyCapacityAndTtlUpdateFromNacos() throws Exception {
        com.alibaba.nacos.api.config.ConfigService client =
                mock(com.alibaba.nacos.api.config.ConfigService.class);
        ArgumentCaptor<Listener> listenerCaptor = ArgumentCaptor.forClass(Listener.class);
        when(client.getConfig(
                org.mockito.ArgumentMatchers.eq("flexlb-test"),
                org.mockito.ArgumentMatchers.eq("FLEXLB_GROUP"),
                org.mockito.ArgumentMatchers.eq(3000L)))
                .thenReturn(localStandbyConfig());
        NacosConfigSource source = createSource(client, "test-namespace");

        source.initialize();
        verify(client).addListener(
                org.mockito.ArgumentMatchers.eq("flexlb-test"),
                org.mockito.ArgumentMatchers.eq("FLEXLB_GROUP"),
                listenerCaptor.capture());
        ConfigService configService = new ConfigService();
        AtomicReference<LocalStandbyRuntimeSettings> runtimeSettings = subscribeToLocalStandbySettings(configService);

        listenerCaptor.getValue().receiveConfigInfo("""
                {"modelServiceConfig":{"kvcm":{"local_standby":{
                "maximum_entries":500,"capacity_multiplier":2.0,
                "ttl_ms":150,"minimum_ttl_ms":20,"ttl_reduction_start_ratio":0.7}}}}
                """);

        assertThat(runtimeSettings.get())
                .isEqualTo(new LocalStandbyRuntimeSettings(500, 2.0, 150, 20, 0.7));
        assertThat(configService.loadBalanceConfig().getModelServiceConfig().getServiceId())
                .isEqualTo("local-standby-service");
        configService.close();
    }

    @Test
    void rejectsInvalidNestedLocalStandbyUpdateAndKeepsLastKnownGoodSnapshot() throws Exception {
        com.alibaba.nacos.api.config.ConfigService client =
                mock(com.alibaba.nacos.api.config.ConfigService.class);
        ArgumentCaptor<Listener> listenerCaptor = ArgumentCaptor.forClass(Listener.class);
        when(client.getConfig(
                org.mockito.ArgumentMatchers.eq("flexlb-test"),
                org.mockito.ArgumentMatchers.eq("FLEXLB_GROUP"),
                org.mockito.ArgumentMatchers.eq(3000L)))
                .thenReturn(localStandbyConfig());
        NacosConfigSource source = createSource(client, "test-namespace");

        source.initialize();
        verify(client).addListener(
                org.mockito.ArgumentMatchers.eq("flexlb-test"),
                org.mockito.ArgumentMatchers.eq("FLEXLB_GROUP"),
                listenerCaptor.capture());
        ConfigService configService = new ConfigService();
        AtomicReference<LocalStandbyRuntimeSettings> runtimeSettings = subscribeToLocalStandbySettings(configService);
        FlexlbConfig lastKnownGood = configService.loadBalanceConfig();
        LocalStandbyRuntimeSettings initialSettings = runtimeSettings.get();

        listenerCaptor.getValue().receiveConfigInfo("""
                {"modelServiceConfig":{"kvcm":{"local_standby":{
                "ttl_ms":100,"minimum_ttl_ms":200}}}}
                """);

        assertThat(configService.loadBalanceConfig()).isSameAs(lastKnownGood);
        assertThat(runtimeSettings.get()).isEqualTo(initialSettings);

        listenerCaptor.getValue().receiveConfigInfo("""
                {"modelServiceConfig":{"kvcm":{"local_standby":{"maximum_entries":500}}}}
                """);

        assertThat(configService.loadBalanceConfig()).isNotSameAs(lastKnownGood);
        assertThat(runtimeSettings.get())
                .isEqualTo(new LocalStandbyRuntimeSettings(500, 10.0, 300_000, 100_000, 0.8));
        configService.close();
    }

    @Test
    void appliesLocalStandbyUpdateWhenKvcmServiceIdChanges() throws Exception {
        com.alibaba.nacos.api.config.ConfigService client =
                mock(com.alibaba.nacos.api.config.ConfigService.class);
        ArgumentCaptor<Listener> listenerCaptor = ArgumentCaptor.forClass(Listener.class);
        when(client.getConfig(
                org.mockito.ArgumentMatchers.eq("flexlb-test"),
                org.mockito.ArgumentMatchers.eq("FLEXLB_GROUP"),
                org.mockito.ArgumentMatchers.eq(3000L)))
                .thenReturn(localStandbyConfig());
        NacosConfigSource source = createSource(client, "test-namespace");

        source.initialize();
        verify(client).addListener(
                org.mockito.ArgumentMatchers.eq("flexlb-test"),
                org.mockito.ArgumentMatchers.eq("FLEXLB_GROUP"),
                listenerCaptor.capture());
        ConfigService configService = new ConfigService();
        AtomicReference<LocalStandbyRuntimeSettings> runtimeSettings = subscribeToLocalStandbySettings(configService);

        listenerCaptor.getValue().receiveConfigInfo("""
                {"modelServiceConfig":{"service_id":"updated-kvcm-service",
                "kvcm":{"local_standby":{"maximum_entries":500,"capacity_multiplier":2.0,
                "ttl_ms":150,"minimum_ttl_ms":20,"ttl_reduction_start_ratio":0.7}}}}
                """);

        assertThat(configService.loadBalanceConfig().getModelServiceConfig().getServiceId())
                .isEqualTo("updated-kvcm-service");
        assertThat(runtimeSettings.get())
                .isEqualTo(new LocalStandbyRuntimeSettings(500, 2.0, 150, 20, 0.7));
        configService.close();
    }

    @Test
    void shutsDownClientWhenRemovingListenerFails() throws Exception {
        com.alibaba.nacos.api.config.ConfigService client =
                mock(com.alibaba.nacos.api.config.ConfigService.class);
        when(client.getConfig(
                org.mockito.ArgumentMatchers.eq("flexlb-test"),
                org.mockito.ArgumentMatchers.eq("FLEXLB_GROUP"),
                org.mockito.ArgumentMatchers.eq(3000L)))
                .thenReturn("{\"maxRetryCount\":9}");
        NacosConfigSource source = createSource(client, "");
        source.initialize();
        ConfigService configService = new ConfigService();
        doThrow(new RuntimeException("remove failed"))
                .when(client)
                .removeListener(
                        org.mockito.ArgumentMatchers.eq("flexlb-test"),
                        org.mockito.ArgumentMatchers.eq("FLEXLB_GROUP"),
                        org.mockito.ArgumentMatchers.any(Listener.class));

        assertThatThrownBy(source::close).hasMessage("remove failed");

        verify(client).shutDown();
        configService.close();
    }

    private NacosConfigSource createSource(
            com.alibaba.nacos.api.config.ConfigService client,
            String namespace) throws Exception {
        NacosConfigSource source = new EnvironmentVariables(
                NACOS_SERVER_ADDR, "127.0.0.1:8848",
                NACOS_DATA_ID, "flexlb-test",
                NACOS_GROUP, "FLEXLB_GROUP",
                NACOS_NAMESPACE, namespace,
                HIPPO_ROLE, "flexlb-test")
                .execute(() -> new NacosConfigSource(new DeploymentIdentity()));
        ReflectionTestUtils.setField(source, "client", client);
        return source;
    }

    private AtomicReference<LocalStandbyRuntimeSettings> subscribeToLocalStandbySettings(ConfigService configService) {
        AtomicReference<LocalStandbyRuntimeSettings> settings = new AtomicReference<>();
        configService.addUpdateListener(LocalStandbyRuntimeSettings::from, settings::set);
        return settings;
    }

    private String localStandbyConfig() {
        return """
                {"modelServiceConfig":{"service_id":"local-standby-service","role_endpoints":[],
                "kvcm":{"enabled":true,"local_standby":{"auto_switch":true,
                "ttl_ms":300000,"minimum_ttl_ms":100000,"ttl_reduction_start_ratio":0.8,
                "maximum_entries":2000,"capacity_multiplier":10.0,
                "async_queue_capacity":100000,"hash_thread_count":4,"hash_queue_capacity":100000}}}}
                """;
    }
}
