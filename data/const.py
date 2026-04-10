# Define Constants for standard OCEL columns
TIME_COL = "time"
EVENT_ID_COL = "event_id"
OBJECT_ID_COL = "object_id"
EVENT_TYPE_COL = "type"
OBJECT_TYPE_COL = "type"
QUALIFIER_COL = "qualifier"
E2O_OBJECT_ID_COL = "object_id"
E2O_EVENT_ID_COL = "event_id"
O2O_DST_COL = "object_id_dst"
O2O_SRC_COL = "object_id_src"

# Define constants for standard OCEL / RelBench table names
EVENT_TABLE = "event"
OBJECT_TABLE = "object"
E2O_TABLE = "e2o"
O2O_TABLE = "o2o"

# Attribute tables are keyed by type, materialized as `f"{PREFIX}{type_name}"`
EVENT_ATTR_TABLE_PREFIX = "event_attr_"
OBJECT_ATTR_TABLE_PREFIX = "object_attr_"
