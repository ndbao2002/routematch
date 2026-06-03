# Topic Matrix

| Topic | Producer | Consumers | Payload Schema |
| --- | --- | --- | --- |
| `Orders_Requested` | Gateway, Dispatch retries | ML Batch Service | `v1/order-requested.schema.json` |
| `Matched_Assignments` | ML Batch Service | Dispatch Engine, Gateway sync worker | `v1/matched-assignments.schema.json` |
| `Offers_Created` | Dispatch Engine | Driver app, Gateway sync worker | `v1/offers-created.schema.json` |
| `Driver_Responses` | Driver app API / Gateway | Dispatch Engine | `v1/driver-responses.schema.json` |
| `Orders_Finalized` | Dispatch Engine | Gateway sync worker | `v1/orders-finalized.schema.json` |
| `Orders_Failed` | Dispatch Engine | Gateway sync worker | `v1/orders-failed.schema.json` |

Envelope:
- Common metadata is defined in `v1/envelope.schema.json`.
- Topic payloads are intended to be wrapped in that envelope when published over Kafka.
