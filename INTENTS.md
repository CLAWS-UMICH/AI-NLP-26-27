# Intents

36 labels (case-sensitive). Intents marked * take a name slot.

| Intent | Description |
|---|---|
| `get_oxy_pri_storage` | Primary oxygen storage level percentage |
| `get_oxy_sec_storage` | Secondary oxygen storage level percentage |
| `get_oxy_pri_pressure` | Primary oxygen tank pressure percentage |
| `get_oxy_sec_pressure` | Secondary oxygen tank pressure percentage |
| `get_suit_pressure_oxy` | Suit oxygen partial pressure reading |
| `get_suit_pressure_co2` | Suit carbon dioxide partial pressure reading |
| `get_suit_pressure_other` | Suit trace gas pressure reading |
| `get_suit_pressure_total` | Total suit pressure reading |
| `get_helmet_pressure_co2` | Helmet carbon dioxide pressure reading |
| `get_fan_pri_rpm` | Primary circulation fan speed in RPM |
| `get_fan_sec_rpm` | Secondary circulation fan speed in RPM |
| `get_scrubber_a_co2_storage` | Scrubber A carbon dioxide storage level |
| `get_scrubber_b_co2_storage` | Scrubber B carbon dioxide storage level |
| `get_temperature` | Suit internal temperature reading |
| `get_coolant_storage` | Coolant storage level percentage |
| `get_coolant_gas_pressure` | Coolant gas loop pressure reading |
| `get_coolant_liquid_pressure` | Coolant liquid loop pressure reading |
| `get_heart_rate` | Astronaut heart rate reading |
| `get_oxy_consumption` | Oxygen consumption rate |
| `get_co2_production` | Carbon dioxide production rate |
| `get_eva_elapsed_time` | Elapsed mission time reading |
| `get_primary_battery_level` | Primary battery level percentage |
| `get_secondary_battery_level` | Secondary battery level percentage |
| `get_warnings` | Active EVA warning and alert summary |
| `open_menu_vitals` | Open EVA vitals HUD panel |
| `open_menu_navigation` | Open EVA navigation and map HUD panel |
| `open_menu_tasks` | Open EVA mission tasks and checklist panel |
| `open_menu_uia` | Open EVA UIA airlock interface panel |
| `close_menu` | Dismiss the active EVA HUD window |
| `Get_coordinates` * | Named target coordinate lookup |
| `Set_navigation_target` * | Set named EVA navigation destination |
| `reroute_navigation` | Calculate a new EVA navigation route |
| `Add_waypoint` * | Add named EVA map waypoint |
| `Delete_waypoint` * | Delete named EVA map waypoint |
| `Complete_task` * | Mark named EVA checklist task complete |
| `unhandled` | Not an EVA command: small talk, noise, off-topic speech |
