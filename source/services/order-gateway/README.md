# Order Gateway Service

Stateless HTTP edge service for order submission and status polling.
Owns Redis status cache updates via the sync worker and publishes OrderRequested events.
