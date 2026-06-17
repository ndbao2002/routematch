package kafka

import (
	"context"
	"encoding/json"
	"time"

	"github.com/google/uuid"
	"github.com/segmentio/kafka-go"
)

type MatchedAssignment struct {
	OrderID      string  `json:"order_id"`
	DriverID     string  `json:"driver_id"`
	Score        float64 `json:"score"`
	AttemptCount int     `json:"attempt_count"`
	Rank         int     `json:"rank"`
	
	// Driver profile features that we can capture from driver-state to populate interaction_logs
	DriverLat      float64 `json:"driver_lat"`
	DriverLon      float64 `json:"driver_lon"`
	DriverDistance float64 `json:"driver_distance_to_pickup"`
	DriverFatigue  float64 `json:"driver_fatigue_index"`
}

type MatchedAssignmentsBatch struct {
	BatchID     string              `json:"batch_id"`
	Assignments []MatchedAssignment `json:"assignments"`
}

type DriverResponsePayload struct {
	OrderID  string `json:"order_id"`
	DriverID string `json:"driver_id"`
	Response string `json:"response"` // "accept" or "reject"
}

type DriverResponseEvent struct {
	EventID       string                `json:"event_id"`
	EventType     string                `json:"event_type"`
	CorrelationID string                `json:"correlation_id"`
	Payload       DriverResponsePayload `json:"payload"`
}

type EventEnvelope struct {
	EventID       string      `json:"event_id"`
	EventType     string      `json:"event_type"`
	EventVersion  int         `json:"event_version"`
	OccurredAt    string      `json:"occurred_at"`
	CorrelationID string      `json:"correlation_id"`
	Payload       interface{} `json:"payload"`
}

type KafkaClient struct {
	brokers []string
	writer  *kafka.Writer
}

func NewKafkaClient(brokers []string) *KafkaClient {
	writer := &kafka.Writer{
		Addr:         kafka.TCP(brokers...),
		Balancer:     &kafka.LeastBytes{},
		RequiredAcks: kafka.RequireAll,
		MaxAttempts:  5,
	}
	return &KafkaClient{
		brokers: brokers,
		writer:  writer,
	}
}

func (c *KafkaClient) Close() error {
	return c.writer.Close()
}

func (c *KafkaClient) PublishEvent(ctx context.Context, topic string, correlationID string, payload interface{}) error {
	envelope := EventEnvelope{
		EventID:       uuid.New().String(),
		EventType:     topic,
		EventVersion:  1,
		OccurredAt:    time.Now().UTC().Format(time.RFC3339),
		CorrelationID: correlationID,
		Payload:       payload,
	}

	bytes, err := json.Marshal(envelope)
	if err != nil {
		return err
	}

	return c.writer.WriteMessages(ctx, kafka.Message{
		Topic: topic,
		Value: bytes,
	})
}

func (c *KafkaClient) NewReader(topic, groupID string) *kafka.Reader {
	return kafka.NewReader(kafka.ReaderConfig{
		Brokers:  c.brokers,
		GroupID:  groupID,
		Topic:    topic,
		MinBytes: 10e3, // 10KB
		MaxBytes: 10e6, // 10MB
		MaxWait:  1 * time.Second,
	})
}
