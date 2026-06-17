package redis

import (
	"context"
	"fmt"
	"time"

	"github.com/redis/go-redis/v9"
)

type RedisClient struct {
	rdb *redis.Client
}

func NewRedisClient(addr string) *RedisClient {
	rdb := redis.NewClient(&redis.Options{
		Addr: addr,
	})
	return &RedisClient{rdb: rdb}
}

func (c *RedisClient) Ping(ctx context.Context) error {
	return c.rdb.Ping(ctx).Err()
}

func (c *RedisClient) AcquireDriverLock(ctx context.Context, driverID, orderID string) (bool, error) {
	key := fmt.Sprintf("lock:driver:%s", driverID)
	value := fmt.Sprintf("order:%s", orderID)
	// Try to acquire lock with 20 seconds expiration
	success, err := c.rdb.SetNX(ctx, key, value, 20*time.Second).Result()
	return success, err
}

func (c *RedisClient) ReleaseDriverLock(ctx context.Context, driverID string) error {
	key := fmt.Sprintf("lock:driver:%s", driverID)
	return c.rdb.Del(ctx, key).Err()
}

func (c *RedisClient) AddToRejectedDrivers(ctx context.Context, orderID, driverID string) error {
	key := fmt.Sprintf("order:rejected:%s", orderID)
	pipe := c.rdb.Pipeline()
	pipe.SAdd(ctx, key, driverID)
	pipe.Expire(ctx, key, 300*time.Second) // 5 minutes TTL
	_, err := pipe.Exec(ctx)
	return err
}
