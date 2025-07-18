"""
Database Connection Pool - PPuRI-AI Ultimate 데이터베이스 연결 풀링
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
import os

# Optional imports with fallbacks
try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    from neo4j import AsyncGraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

try:
    from motor.motor_asyncio import AsyncIOMotorClient
    MOTOR_AVAILABLE = True
except ImportError:
    MOTOR_AVAILABLE = False

from ..constants import DatabaseConstants
from ..models.exceptions import DatabaseException, CircuitBreakerException

logger = logging.getLogger(__name__)


class ConnectionPoolManager:
    """데이터베이스 연결 풀 관리자"""
    
    def __init__(self):
        self.postgres_pool: Optional[Any] = None
        self.redis_pool: Optional[Any] = None
        self.neo4j_driver: Optional[Any] = None
        self.mongo_client: Optional[Any] = None
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
    async def initialize(self):
        """모든 데이터베이스 연결 풀 초기화"""
        success_count = 0
        total_count = 0
        
        # PostgreSQL 연결 풀 초기화
        if ASYNCPG_AVAILABLE:
            total_count += 1
            try:
                await self._init_postgres_pool()
                success_count += 1
                logger.info("PostgreSQL 연결 풀 초기화 성공")
            except Exception as e:
                logger.error(f"PostgreSQL 연결 풀 초기화 실패: {e}")
                logger.info("PostgreSQL 없이 서버 계속 실행")
        else:
            logger.warning("asyncpg를 사용할 수 없습니다. PostgreSQL 연결 풀을 건너뜁니다.")
        
        # Redis 연결 풀 초기화
        if REDIS_AVAILABLE:
            total_count += 1
            try:
                await self._init_redis_pool()
                success_count += 1
                logger.info("Redis 연결 풀 초기화 성공")
            except Exception as e:
                logger.error(f"Redis 연결 풀 초기화 실패: {e}")
                logger.info("Redis 없이 서버 계속 실행")
        else:
            logger.warning("redis를 사용할 수 없습니다. Redis 연결 풀을 건너뜁니다.")
        
        # Neo4j 드라이버 초기화
        if NEO4J_AVAILABLE:
            total_count += 1
            try:
                await self._init_neo4j_driver()
                success_count += 1
                logger.info("Neo4j 드라이버 초기화 성공")
            except Exception as e:
                logger.error(f"Neo4j 드라이버 초기화 실패: {e}")
                logger.info("Neo4j 없이 서버 계속 실행")
        else:
            logger.warning("neo4j를 사용할 수 없습니다. Neo4j 드라이버를 건너뜁니다.")
        
        # MongoDB 클라이언트 초기화 (선택사항)
        # if MOTOR_AVAILABLE:
        #     await self._init_mongo_client()
        
        # Circuit Breaker 초기화
        self._init_circuit_breakers()
        
        logger.info(f"데이터베이스 연결 풀 초기화 완료 ({success_count}/{total_count} 성공)")
        
        # 모든 연결에 실패하면 경고하지만 서버는 계속 실행
        if total_count > 0 and success_count == 0:
            logger.warning("모든 데이터베이스 연결에 실패했습니다. 제한된 기능으로 서버를 실행합니다.")
        elif total_count == 0:
            logger.warning("사용 가능한 데이터베이스 드라이버가 없습니다. 제한된 기능으로 서버를 실행합니다.")
    
    async def _init_postgres_pool(self):
        """PostgreSQL 연결 풀 초기화"""
        try:
            database_url = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/ppuri_ai")
            
            self.postgres_pool = await asyncpg.create_pool(
                database_url,
                min_size=DatabaseConstants.POSTGRES_MIN_CONNECTIONS,
                max_size=DatabaseConstants.POSTGRES_MAX_CONNECTIONS,
                max_queries=DatabaseConstants.POSTGRES_MAX_QUERIES,
                max_inactive_connection_lifetime=DatabaseConstants.POSTGRES_CONNECTION_TIMEOUT,
                command_timeout=30,
                server_settings={
                    'jit': 'off',
                    'application_name': 'ppuri-ai-ultimate'
                }
            )
            
            logger.info("PostgreSQL 연결 풀 초기화 완료")
            
        except Exception as e:
            logger.error(f"PostgreSQL 연결 풀 초기화 실패: {e}")
            raise DatabaseException("PostgreSQL 연결 풀 초기화 실패")
    
    async def _init_redis_pool(self):
        """Redis 연결 풀 초기화"""
        try:
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
            
            self.redis_pool = redis.ConnectionPool.from_url(
                redis_url,
                max_connections=DatabaseConstants.REDIS_MAX_CONNECTIONS,
                socket_connect_timeout=DatabaseConstants.REDIS_CONNECTION_TIMEOUT,
                socket_timeout=DatabaseConstants.REDIS_SOCKET_TIMEOUT,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # 연결 테스트
            redis_client = redis.Redis(connection_pool=self.redis_pool)
            await redis_client.ping()
            await redis_client.close()
            
            logger.info("Redis 연결 풀 초기화 완료")
            
        except Exception as e:
            logger.error(f"Redis 연결 풀 초기화 실패: {e}")
            raise DatabaseException("Redis 연결 풀 초기화 실패")
    
    async def _init_neo4j_driver(self):
        """Neo4j 드라이버 초기화"""
        try:
            neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
            neo4j_user = os.getenv("NEO4J_USER", "neo4j")
            neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
            
            self.neo4j_driver = AsyncGraphDatabase.driver(
                neo4j_uri,
                auth=(neo4j_user, neo4j_password),
                max_connection_pool_size=DatabaseConstants.NEO4J_MAX_CONNECTION_POOL_SIZE,
                connection_acquisition_timeout=DatabaseConstants.NEO4J_CONNECTION_ACQUISITION_TIMEOUT,
                max_transaction_retry_time=DatabaseConstants.NEO4J_MAX_TRANSACTION_RETRY_TIME,
                keep_alive=True
            )
            
            # 연결 테스트
            await self.neo4j_driver.verify_connectivity()
            
            logger.info("Neo4j 드라이버 초기화 완료")
            
        except Exception as e:
            logger.error(f"Neo4j 드라이버 초기화 실패: {e}")
            raise DatabaseException("Neo4j 드라이버 초기화 실패")
    
    def _init_circuit_breakers(self):
        """Circuit Breaker 초기화"""
        self.circuit_breakers = {
            "postgres": CircuitBreaker(
                failure_threshold=5,
                recovery_timeout=30,
                expected_exception=DatabaseException
            ),
            "redis": CircuitBreaker(
                failure_threshold=3,
                recovery_timeout=20,
                expected_exception=DatabaseException
            ),
            "neo4j": CircuitBreaker(
                failure_threshold=5,
                recovery_timeout=30,
                expected_exception=DatabaseException
            )
        }
    
    @asynccontextmanager
    async def get_postgres_connection(self):
        """PostgreSQL 연결 컨텍스트 매니저"""
        if not self.postgres_pool:
            raise DatabaseException("PostgreSQL 연결 풀이 초기화되지 않았습니다.")
        
        circuit_breaker = self.circuit_breakers.get("postgres")
        if circuit_breaker and circuit_breaker.is_open():
            raise CircuitBreakerException("PostgreSQL 서비스 일시 중단")
        
        connection = None
        try:
            connection = await self.postgres_pool.acquire()
            yield connection
            
            # Circuit Breaker 성공 기록
            if circuit_breaker:
                circuit_breaker.record_success()
                
        except Exception as e:
            # Circuit Breaker 실패 기록
            if circuit_breaker:
                circuit_breaker.record_failure()
            
            logger.error(f"PostgreSQL 연결 오류: {e}")
            raise DatabaseException("PostgreSQL 연결 오류")
        finally:
            if connection:
                await self.postgres_pool.release(connection)
    
    @asynccontextmanager
    async def get_redis_connection(self):
        """Redis 연결 컨텍스트 매니저"""
        if not self.redis_pool:
            raise DatabaseException("Redis 연결 풀이 초기화되지 않았습니다.")
        
        circuit_breaker = self.circuit_breakers.get("redis")
        if circuit_breaker and circuit_breaker.is_open():
            raise CircuitBreakerException("Redis 서비스 일시 중단")
        
        client = None
        try:
            client = redis.Redis(connection_pool=self.redis_pool)
            yield client
            
            # Circuit Breaker 성공 기록
            if circuit_breaker:
                circuit_breaker.record_success()
                
        except Exception as e:
            # Circuit Breaker 실패 기록
            if circuit_breaker:
                circuit_breaker.record_failure()
            
            logger.error(f"Redis 연결 오류: {e}")
            raise DatabaseException("Redis 연결 오류")
        finally:
            if client:
                await client.close()
    
    @asynccontextmanager
    async def get_neo4j_session(self, database: str = "neo4j"):
        """Neo4j 세션 컨텍스트 매니저"""
        if not self.neo4j_driver:
            raise DatabaseException("Neo4j 드라이버가 초기화되지 않았습니다.")
        
        circuit_breaker = self.circuit_breakers.get("neo4j")
        if circuit_breaker and circuit_breaker.is_open():
            raise CircuitBreakerException("Neo4j 서비스 일시 중단")
        
        session = None
        try:
            session = self.neo4j_driver.session(database=database)
            yield session
            
            # Circuit Breaker 성공 기록
            if circuit_breaker:
                circuit_breaker.record_success()
                
        except Exception as e:
            # Circuit Breaker 실패 기록
            if circuit_breaker:
                circuit_breaker.record_failure()
            
            logger.error(f"Neo4j 연결 오류: {e}")
            raise DatabaseException("Neo4j 연결 오류")
        finally:
            if session:
                await session.close()
    
    async def close_all(self):
        """모든 연결 풀 종료"""
        try:
            if self.postgres_pool:
                await self.postgres_pool.close()
            
            if self.redis_pool:
                await self.redis_pool.disconnect()
            
            if self.neo4j_driver:
                await self.neo4j_driver.close()
            
            if self.mongo_client:
                self.mongo_client.close()
            
            logger.info("모든 데이터베이스 연결 풀 종료 완료")
            
        except Exception as e:
            logger.error(f"데이터베이스 연결 풀 종료 실패: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """연결 풀 상태 확인"""
        status = {
            "postgres": {"status": "not_configured", "pool_size": 0, "available": 0},
            "redis": {"status": "not_configured", "connections": 0},
            "neo4j": {"status": "not_configured", "available": True}
        }
        
        # PostgreSQL 상태 확인
        if self.postgres_pool:
            try:
                async with self.get_postgres_connection() as conn:
                    await conn.fetchval("SELECT 1")
                status["postgres"]["status"] = "healthy"
                status["postgres"]["pool_size"] = self.postgres_pool.get_size()
                status["postgres"]["available"] = self.postgres_pool.get_idle_size()
            except Exception as e:
                status["postgres"]["status"] = f"error: {e}"
        elif ASYNCPG_AVAILABLE:
            status["postgres"]["status"] = "connection_failed"
        else:
            status["postgres"]["status"] = "driver_not_available"
        
        # Redis 상태 확인
        if self.redis_pool:
            try:
                async with self.get_redis_connection() as client:
                    await client.ping()
                status["redis"]["status"] = "healthy"
                status["redis"]["connections"] = self.redis_pool.created_connections
            except Exception as e:
                status["redis"]["status"] = f"error: {e}"
        elif REDIS_AVAILABLE:
            status["redis"]["status"] = "connection_failed"
        else:
            status["redis"]["status"] = "driver_not_available"
        
        # Neo4j 상태 확인
        if self.neo4j_driver:
            try:
                async with self.get_neo4j_session() as session:
                    await session.run("RETURN 1")
                status["neo4j"]["status"] = "healthy"
            except Exception as e:
                status["neo4j"]["status"] = f"error: {e}"
        elif NEO4J_AVAILABLE:
            status["neo4j"]["status"] = "connection_failed"
        else:
            status["neo4j"]["status"] = "driver_not_available"
        
        return status


class CircuitBreaker:
    """Circuit Breaker 패턴 구현"""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 30,
        expected_exception: Exception = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def is_open(self) -> bool:
        """Circuit Breaker가 열려있는지 확인"""
        if self.state == "OPEN":
            # 복구 시간이 지났는지 확인
            if self.last_failure_time and (
                datetime.now() - self.last_failure_time
            ).total_seconds() > self.recovery_timeout:
                self.state = "HALF_OPEN"
                return False
            return True
        return False
    
    def record_success(self):
        """성공 기록"""
        self.failure_count = 0
        self.state = "CLOSED"
        self.last_failure_time = None
    
    def record_failure(self):
        """실패 기록"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"


# 전역 연결 풀 관리자 인스턴스
connection_pool_manager = ConnectionPoolManager()