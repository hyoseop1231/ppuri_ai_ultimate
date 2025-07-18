"""
Database Optimization - PPuRI-AI Ultimate 데이터베이스 최적화 스크립트
"""

import asyncio
import logging
from typing import List, Dict, Any
from datetime import datetime

from api.database.connection_pool import connection_pool_manager
from api.constants import DatabaseConstants

logger = logging.getLogger(__name__)


class DatabaseOptimizer:
    """데이터베이스 최적화 관리자"""
    
    def __init__(self):
        self.pool_manager = connection_pool_manager
    
    async def optimize_all_databases(self):
        """모든 데이터베이스 최적화 실행"""
        logger.info("데이터베이스 최적화 시작")
        
        try:
            # PostgreSQL 최적화
            await self.optimize_postgresql()
            
            # Neo4j 최적화
            await self.optimize_neo4j()
            
            # Redis 최적화
            await self.optimize_redis()
            
            logger.info("모든 데이터베이스 최적화 완료")
            
        except Exception as e:
            logger.error(f"데이터베이스 최적화 실패: {e}")
            raise
    
    async def optimize_postgresql(self):
        """PostgreSQL 최적화"""
        logger.info("PostgreSQL 최적화 시작")
        
        try:
            async with self.pool_manager.get_postgres_connection() as conn:
                # 기본 테이블 생성
                await self._create_postgresql_tables(conn)
                
                # 인덱스 생성
                await self._create_postgresql_indexes(conn)
                
                # 뷰 생성
                await self._create_postgresql_views(conn)
                
                # 성능 설정
                await self._configure_postgresql_performance(conn)
                
            logger.info("PostgreSQL 최적화 완료")
            
        except Exception as e:
            logger.error(f"PostgreSQL 최적화 실패: {e}")
            raise
    
    async def _create_postgresql_tables(self, conn):
        """PostgreSQL 테이블 생성"""
        tables = [
            # 사용자 테이블
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id VARCHAR(255) PRIMARY KEY,
                username VARCHAR(100) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                role VARCHAR(50) DEFAULT 'user',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT true
            )
            """,
            
            # 세션 테이블
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id VARCHAR(255) PRIMARY KEY,
                user_id VARCHAR(255) REFERENCES users(user_id),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT true,
                metadata JSONB
            )
            """,
            
            # 메시지 테이블
            """
            CREATE TABLE IF NOT EXISTS messages (
                message_id VARCHAR(255) PRIMARY KEY,
                session_id VARCHAR(255) REFERENCES sessions(session_id),
                user_id VARCHAR(255) REFERENCES users(user_id),
                content TEXT NOT NULL,
                message_type VARCHAR(50) DEFAULT 'text',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSONB
            )
            """,
            
            # AI 응답 테이블
            """
            CREATE TABLE IF NOT EXISTS ai_responses (
                response_id VARCHAR(255) PRIMARY KEY,
                message_id VARCHAR(255) REFERENCES messages(message_id),
                response_text TEXT NOT NULL,
                model_used VARCHAR(100),
                processing_time DECIMAL(10, 3),
                confidence DECIMAL(5, 3),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSONB
            )
            """,
            
            # 성능 메트릭 테이블
            """
            CREATE TABLE IF NOT EXISTS performance_metrics (
                metric_id SERIAL PRIMARY KEY,
                metric_name VARCHAR(100) NOT NULL,
                metric_value DECIMAL(15, 6) NOT NULL,
                component VARCHAR(100),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSONB
            )
            """
        ]
        
        for table_sql in tables:
            await conn.execute(table_sql)
            logger.info(f"테이블 생성/확인 완료")
    
    async def _create_postgresql_indexes(self, conn):
        """PostgreSQL 인덱스 생성"""
        indexes = [
            # 사용자 인덱스
            "CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)",
            "CREATE INDEX IF NOT EXISTS idx_users_created_at ON users(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_users_is_active ON users(is_active)",
            
            # 세션 인덱스
            "CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON sessions(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_sessions_last_activity ON sessions(last_activity)",
            "CREATE INDEX IF NOT EXISTS idx_sessions_is_active ON sessions(is_active)",
            
            # 메시지 인덱스
            "CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_messages_user_id ON messages(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_messages_message_type ON messages(message_type)",
            
            # AI 응답 인덱스
            "CREATE INDEX IF NOT EXISTS idx_ai_responses_message_id ON ai_responses(message_id)",
            "CREATE INDEX IF NOT EXISTS idx_ai_responses_model_used ON ai_responses(model_used)",
            "CREATE INDEX IF NOT EXISTS idx_ai_responses_created_at ON ai_responses(created_at)",
            
            # 성능 메트릭 인덱스
            "CREATE INDEX IF NOT EXISTS idx_performance_metrics_name ON performance_metrics(metric_name)",
            "CREATE INDEX IF NOT EXISTS idx_performance_metrics_component ON performance_metrics(component)",
            "CREATE INDEX IF NOT EXISTS idx_performance_metrics_timestamp ON performance_metrics(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_performance_metrics_name_timestamp ON performance_metrics(metric_name, timestamp)",
            
            # 복합 인덱스
            "CREATE INDEX IF NOT EXISTS idx_messages_session_created ON messages(session_id, created_at)",
            "CREATE INDEX IF NOT EXISTS idx_sessions_user_activity ON sessions(user_id, last_activity)",
            
            # JSONB 인덱스
            "CREATE INDEX IF NOT EXISTS idx_sessions_metadata_gin ON sessions USING GIN(metadata)",
            "CREATE INDEX IF NOT EXISTS idx_messages_metadata_gin ON messages USING GIN(metadata)",
        ]
        
        for index_sql in indexes:
            await conn.execute(index_sql)
        
        logger.info("PostgreSQL 인덱스 생성 완료")
    
    async def _create_postgresql_views(self, conn):
        """PostgreSQL 뷰 생성"""
        views = [
            # 사용자 활동 뷰
            """
            CREATE OR REPLACE VIEW user_activity_summary AS
            SELECT 
                u.user_id,
                u.username,
                COUNT(s.session_id) as total_sessions,
                COUNT(m.message_id) as total_messages,
                MAX(s.last_activity) as last_activity
            FROM users u
            LEFT JOIN sessions s ON u.user_id = s.user_id
            LEFT JOIN messages m ON s.session_id = m.session_id
            GROUP BY u.user_id, u.username
            """,
            
            # 성능 메트릭 요약 뷰
            """
            CREATE OR REPLACE VIEW performance_summary AS
            SELECT 
                metric_name,
                component,
                AVG(metric_value) as avg_value,
                MIN(metric_value) as min_value,
                MAX(metric_value) as max_value,
                COUNT(*) as sample_count,
                MAX(timestamp) as last_updated
            FROM performance_metrics
            WHERE timestamp >= NOW() - INTERVAL '1 hour'
            GROUP BY metric_name, component
            """
        ]
        
        for view_sql in views:
            await conn.execute(view_sql)
        
        logger.info("PostgreSQL 뷰 생성 완료")
    
    async def _configure_postgresql_performance(self, conn):
        """PostgreSQL 성능 설정"""
        # 데이터베이스 설정은 실제 환경에서 ALTER SYSTEM 권한이 필요
        # 여기서는 세션 레벨 설정만 적용
        settings = [
            "SET work_mem = '256MB'",
            "SET maintenance_work_mem = '1GB'",
            "SET effective_cache_size = '4GB'",
            "SET random_page_cost = 1.1",
            "SET seq_page_cost = 1.0"
        ]
        
        for setting in settings:
            try:
                await conn.execute(setting)
            except Exception as e:
                logger.warning(f"성능 설정 실패: {setting}, 오류: {e}")
        
        logger.info("PostgreSQL 성능 설정 완료")
    
    async def optimize_neo4j(self):
        """Neo4j 최적화"""
        logger.info("Neo4j 최적화 시작")
        
        try:
            async with self.pool_manager.get_neo4j_session() as session:
                # 인덱스 생성
                await self._create_neo4j_indexes(session)
                
                # 제약 조건 생성
                await self._create_neo4j_constraints(session)
                
                # 통계 업데이트
                await self._update_neo4j_statistics(session)
                
            logger.info("Neo4j 최적화 완료")
            
        except Exception as e:
            logger.error(f"Neo4j 최적화 실패: {e}")
            raise
    
    async def _create_neo4j_indexes(self, session):
        """Neo4j 인덱스 생성"""
        indexes = [
            # 노드 인덱스
            "CREATE INDEX node_session_idx IF NOT EXISTS FOR (n:Node) ON (n.session_id)",
            "CREATE INDEX node_global_idx IF NOT EXISTS FOR (n:Node) ON (n.global)",
            "CREATE INDEX node_timestamp_idx IF NOT EXISTS FOR (n:Node) ON (n.timestamp)",
            "CREATE INDEX node_type_idx IF NOT EXISTS FOR (n:Node) ON (n.type)",
            
            # 사용자 인덱스
            "CREATE INDEX user_idx IF NOT EXISTS FOR (u:User) ON (u.user_id)",
            "CREATE INDEX user_username_idx IF NOT EXISTS FOR (u:User) ON (u.username)",
            
            # 세션 인덱스
            "CREATE INDEX session_idx IF NOT EXISTS FOR (s:Session) ON (s.session_id)",
            "CREATE INDEX session_user_idx IF NOT EXISTS FOR (s:Session) ON (s.user_id)",
            
            # 관계 인덱스
            "CREATE INDEX relationship_type_idx IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.type)",
            "CREATE INDEX relationship_strength_idx IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.strength)",
            "CREATE INDEX relationship_timestamp_idx IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.timestamp)"
        ]
        
        for index_cypher in indexes:
            await session.run(index_cypher)
        
        logger.info("Neo4j 인덱스 생성 완료")
    
    async def _create_neo4j_constraints(self, session):
        """Neo4j 제약 조건 생성"""
        constraints = [
            # 유니크 제약 조건
            "CREATE CONSTRAINT user_id_unique IF NOT EXISTS FOR (u:User) REQUIRE u.user_id IS UNIQUE",
            "CREATE CONSTRAINT session_id_unique IF NOT EXISTS FOR (s:Session) REQUIRE s.session_id IS UNIQUE",
            "CREATE CONSTRAINT node_id_unique IF NOT EXISTS FOR (n:Node) REQUIRE n.id IS UNIQUE",
            
            # 존재 제약 조건
            "CREATE CONSTRAINT user_id_exists IF NOT EXISTS FOR (u:User) REQUIRE u.user_id IS NOT NULL",
            "CREATE CONSTRAINT session_id_exists IF NOT EXISTS FOR (s:Session) REQUIRE s.session_id IS NOT NULL"
        ]
        
        for constraint_cypher in constraints:
            try:
                await session.run(constraint_cypher)
            except Exception as e:
                logger.warning(f"제약 조건 생성 실패: {constraint_cypher}, 오류: {e}")
        
        logger.info("Neo4j 제약 조건 생성 완료")
    
    async def _update_neo4j_statistics(self, session):
        """Neo4j 통계 업데이트"""
        # 통계 업데이트 쿼리
        await session.run("CALL db.stats.retrieve('GRAPH COUNTS')")
        
        logger.info("Neo4j 통계 업데이트 완료")
    
    async def optimize_redis(self):
        """Redis 최적화"""
        logger.info("Redis 최적화 시작")
        
        try:
            async with self.pool_manager.get_redis_connection() as client:
                # 메모리 최적화
                await self._optimize_redis_memory(client)
                
                # 만료 정책 설정
                await self._configure_redis_expiry(client)
                
                # 정리 작업
                await self._cleanup_redis_data(client)
                
            logger.info("Redis 최적화 완료")
            
        except Exception as e:
            logger.error(f"Redis 최적화 실패: {e}")
            raise
    
    async def _optimize_redis_memory(self, client):
        """Redis 메모리 최적화"""
        # 메모리 사용량 확인
        memory_info = await client.info("memory")
        used_memory = memory_info.get("used_memory", 0)
        
        if used_memory > 100 * 1024 * 1024:  # 100MB 이상
            # 메모리 정리
            await client.flushdb()
            logger.info("Redis 메모리 정리 완료")
    
    async def _configure_redis_expiry(self, client):
        """Redis 만료 정책 설정"""
        # 세션 키 만료 시간 설정
        session_keys = await client.keys("session:*")
        for key in session_keys:
            await client.expire(key, 3600)  # 1시간
        
        # 캐시 키 만료 시간 설정
        cache_keys = await client.keys("cache:*")
        for key in cache_keys:
            await client.expire(key, 1800)  # 30분
        
        logger.info("Redis 만료 정책 설정 완료")
    
    async def _cleanup_redis_data(self, client):
        """Redis 데이터 정리"""
        # 만료된 키 정리
        expired_keys = []
        all_keys = await client.keys("*")
        
        for key in all_keys:
            ttl = await client.ttl(key)
            if ttl == -1:  # 만료 시간이 설정되지 않은 키
                # 패턴에 따라 기본 만료 시간 설정
                if key.startswith("session:"):
                    await client.expire(key, 3600)
                elif key.startswith("cache:"):
                    await client.expire(key, 1800)
                elif key.startswith("temp:"):
                    await client.expire(key, 600)  # 10분
        
        logger.info("Redis 데이터 정리 완료")
    
    async def analyze_performance(self) -> Dict[str, Any]:
        """성능 분석 보고서 생성"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "databases": {}
        }
        
        # PostgreSQL 성능 분석
        try:
            async with self.pool_manager.get_postgres_connection() as conn:
                # 테이블 크기 분석
                table_sizes = await conn.fetch("""
                    SELECT 
                        schemaname,
                        tablename,
                        pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
                    FROM pg_tables 
                    WHERE schemaname NOT IN ('information_schema', 'pg_catalog')
                """)
                
                # 인덱스 사용률 분석
                index_usage = await conn.fetch("""
                    SELECT 
                        indexrelname,
                        idx_tup_read,
                        idx_tup_fetch,
                        idx_scan
                    FROM pg_stat_user_indexes 
                    ORDER BY idx_scan DESC
                """)
                
                report["databases"]["postgresql"] = {
                    "table_sizes": [dict(row) for row in table_sizes],
                    "index_usage": [dict(row) for row in index_usage]
                }
        except Exception as e:
            report["databases"]["postgresql"] = {"error": str(e)}
        
        # Redis 성능 분석
        try:
            async with self.pool_manager.get_redis_connection() as client:
                info = await client.info()
                
                report["databases"]["redis"] = {
                    "memory_usage": info.get("used_memory", 0),
                    "connected_clients": info.get("connected_clients", 0),
                    "total_commands_processed": info.get("total_commands_processed", 0),
                    "keyspace_hits": info.get("keyspace_hits", 0),
                    "keyspace_misses": info.get("keyspace_misses", 0)
                }
        except Exception as e:
            report["databases"]["redis"] = {"error": str(e)}
        
        return report


async def main():
    """최적화 스크립트 실행"""
    logging.basicConfig(level=logging.INFO)
    
    try:
        # 연결 풀 초기화
        await connection_pool_manager.initialize()
        
        # 최적화 실행
        optimizer = DatabaseOptimizer()
        await optimizer.optimize_all_databases()
        
        # 성능 분석 보고서 생성
        report = await optimizer.analyze_performance()
        print("성능 분석 보고서:")
        print(report)
        
    except Exception as e:
        logger.error(f"최적화 스크립트 실행 실패: {e}")
    finally:
        # 연결 풀 정리
        await connection_pool_manager.close_all()


if __name__ == "__main__":
    asyncio.run(main())