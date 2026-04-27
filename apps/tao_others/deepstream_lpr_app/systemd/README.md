Install the service as root on the field unit:

```shell
sudo cp alpr-field-unit.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable alpr-field-unit.service
sudo systemctl start alpr-field-unit.service
```

Useful commands:

```shell
sudo systemctl status alpr-field-unit.service
sudo systemctl restart alpr-field-unit.service
sudo journalctl -u alpr-field-unit.service -f
```

The service wraps the existing `scripts/start_field_unit.sh` and `scripts/stop_field_unit.sh` workflow, so logs and pid files remain under the application directory.

Optional environment file:

The unit now reads `/etc/default/alpr-field-unit` if it exists. Use that file to set the dashboard bind address/port and optional boot-time network helpers.

Start from the example in [systemd/alpr-field-unit.env.example](systemd/alpr-field-unit.env.example) and copy it to `/etc/default/alpr-field-unit` on the field unit.

Example:

```shell
ALPR_DASHBOARD_HOST=0.0.0.0
ALPR_DASHBOARD_PORT=8080

ALPR_WIFI_AP_ENABLE=1
ALPR_WIFI_AP_INTERFACE=wlp3s0
ALPR_WIFI_AP_SSID=ALPR-Field-Unit
ALPR_WIFI_AP_PASSWORD=ALPRAccess123

ALPR_ETHERNET_SHARE_ENABLE=1
ALPR_ETHERNET_INTERFACE=enp2s0
```

The WiFi and Ethernet helpers use `scripts/configure_network_access.sh`, which depends on NetworkManager and `nmcli`.