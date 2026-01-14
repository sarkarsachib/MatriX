"""
Knowledge Store for Direction Mode
SQLite database for caching results and persistent knowledge storage
"""

import sqlite3
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import time
import hashlib
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class CachedQuery:
    """Data class for cached queries"""
    id: int
    query: str
    query_hash: str
    user_id: str
    mode: str
    submode: str
    results: str  # JSON string
    timestamp: float
    confidence: float
    sources: str  # JSON string of sources

class KnowledgeStore:
    """
    SQLite-based knowledge store for caching and persistent storage
    """
    
    def __init__(self, db_path: str = "direction_mode_knowledge.db"):
        """
        Initialize the knowledge store
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._ensure_database()
        logger.info(f"KnowledgeStore initialized with database: {db_path}")
    
    def _ensure_database(self):
        """Create database tables if they don't exist"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create queries cache table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS queries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    query_hash TEXT NOT NULL UNIQUE,
                    user_id TEXT,
                    mode TEXT,
                    submode TEXT,
                    results TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    confidence REAL DEFAULT 0.0,
                    sources TEXT DEFAULT '[]',
                    FOREIGN KEY (id) REFERENCES facts (query_id)
                )
            ''')
            
            # Create facts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_id INTEGER,
                    fact TEXT NOT NULL,
                    source TEXT NOT NULL,
                    source_url TEXT,
                    confidence REAL DEFAULT 0.0,
                    context TEXT,
                    entities TEXT DEFAULT '[]',
                    timestamp REAL NOT NULL,
                    FOREIGN KEY (query_id) REFERENCES queries (id) ON DELETE CASCADE
                )
            ''')
            
            # Create knowledge base table for persistent concepts
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS knowledge_base (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    concept TEXT NOT NULL,
                    definition TEXT,
                    related_facts TEXT DEFAULT '[]',
                    popularity REAL DEFAULT 0.0,
                    last_accessed REAL DEFAULT 0.0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    UNIQUE(concept)
                )
            ''')
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_query_hash ON queries(query_hash)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON queries(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_fact_confidence ON facts(confidence)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_concept ON knowledge_base(concept)')
            
            conn.commit()
            logger.info("Database tables created/verified successfully")
    
    def store_query_result(self, query: str, user_id: str, results: Dict[str, Any], 
                          mode: str = "direction", submode: str = "normal") -> int:
        """
        Store query results in cache
        
        Args:
            query: The original query
            user_id: User identifier
            results: Results dictionary
            mode: Processing mode
            submode: Response sub-mode
            
        Returns:
            ID of the stored query
        """
        query_hash = self._generate_query_hash(query, user_id, mode, submode)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Store main query
            cursor.execute('''
                INSERT OR REPLACE INTO queries 
                (query, query_hash, user_id, mode, submode, results, timestamp, confidence, sources)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                query,
                query_hash,
                user_id,
                mode,
                submode,
                json.dumps(results),
                time.time(),
                results.get('confidence', 0.0),
                json.dumps(results.get('sources', []))
            ))
            
            query_id = cursor.lastrowid
            
            # Store individual facts if available
            if 'facts' in results:
                for fact_data in results['facts']:
                    cursor.execute('''
                        INSERT INTO facts 
                        (query_id, fact, source, source_url, confidence, context, entities, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        query_id,
                        fact_data.get('fact', ''),
                        fact_data.get('source', ''),
                        fact_data.get('source_url', ''),
                        fact_data.get('confidence', 0.0),
                        fact_data.get('context', ''),
                        json.dumps(fact_data.get('entities', [])),
                        time.time()
                    ))
            
            # Update knowledge base with concepts
            if 'key_information' in results:
                self._update_knowledge_base(results['key_information'], query_id)
            
            conn.commit()
            
        logger.info(f"Stored query result with ID: {query_id}")
        return query_id
    
    def retrieve_similar_queries(self, query: str, user_id: str = None, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve similar cached queries
        
        Args:
            query: Query to match against
            user_id: Optional user ID to filter by
            limit: Maximum number of results
            
        Returns:
            List of similar queries with results
        """
        query_hash = self._generate_query_hash(query, user_id or "", "direction", "normal")
        query_words = set(query.lower().split())
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Build WHERE clause
            where_conditions = []
            params = []
            
            # Match on query words (simple similarity)
            word_conditions = []
            for word in query_words:
                if len(word) > 2:  # Ignore very short words
                    word_conditions.append("query LIKE ?")
                    params.append(f"%{word}%")
            
            if word_conditions:
                where_conditions.append("(" + " OR ".join(word_conditions) + ")")
            
            # User filter
            if user_id:
                where_conditions.append("user_id = ?")
                params.append(user_id)
            
            where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
            
            cursor.execute(f'''
                SELECT id, query, results, timestamp, confidence, sources
                FROM queries
                WHERE {where_clause}
                ORDER BY confidence DESC, timestamp DESC
                LIMIT ?
            ''', params + [limit])
            
            results = []
            for row in cursor.fetchall():
                query_id, stored_query, results_json, timestamp, confidence, sources_json = row
                
                try:
                    results_data = json.loads(results_json)
                    sources_data = json.loads(sources_json) if sources_json else []
                    
                    results.append({
                        'query_id': query_id,
                        'query': stored_query,
                        'results': results_data,
                        'timestamp': timestamp,
                        'confidence': confidence,
                        'sources': sources_data,
                        'similarity_score': self._calculate_similarity(query, stored_query)
                    })
                except json.JSONDecodeError as e:
                    logger.warning(f"Error decoding cached results for query {query_id}: {e}")
                    continue
        
        # Sort by similarity score
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        logger.info(f"Retrieved {len(results)} similar queries for: {query[:50]}...")
        return results
    
    def get_facts_for_query(self, query_id: int) -> List[Dict[str, Any]]:
        """
        Retrieve facts for a specific query
        
        Args:
            query_id: Query ID
            
        Returns:
            List of facts with metadata
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, fact, source, source_url, confidence, context, entities, timestamp
                FROM facts
                WHERE query_id = ?
                ORDER BY confidence DESC
            ''', (query_id,))
            
            facts = []
            for row in cursor.fetchall():
                fact_id, fact, source, source_url, confidence, context, entities_json, timestamp = row
                
                try:
                    entities = json.loads(entities_json) if entities_json else []
                    
                    facts.append({
                        'fact_id': fact_id,
                        'fact': fact,
                        'source': source,
                        'source_url': source_url,
                        'confidence': confidence,
                        'context': context,
                        'entities': entities,
                        'timestamp': timestamp
                    })
                except json.JSONDecodeError as e:
                    logger.warning(f"Error decoding entities for fact {fact_id}: {e}")
                    continue
        
        return facts
    
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge base
        
        Returns:
            Dictionary containing statistics
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get query statistics
            cursor.execute('SELECT COUNT(*) FROM queries')
            total_queries = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM facts')
            total_facts = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM knowledge_base')
            total_concepts = cursor.fetchone()[0]
            
            # Get recent activity
            cursor.execute('''
                SELECT COUNT(*) FROM queries 
                WHERE timestamp > ?
            ''', (time.time() - 86400,))  # Last 24 hours
            
            recent_queries = cursor.fetchone()[0]
            
            # Get average confidence
            cursor.execute('SELECT AVG(confidence) FROM queries')
            avg_confidence = cursor.fetchone()[0] or 0.0
            
            # Get top sources
            cursor.execute('''
                SELECT source, COUNT(*) as count
                FROM facts
                GROUP BY source
                ORDER BY count DESC
                LIMIT 5
            ''')
            
            top_sources = [{'source': row[0], 'count': row[1]} for row in cursor.fetchall()]
            
            # Get most popular concepts
            cursor.execute('''
                SELECT concept, popularity, last_accessed
                FROM knowledge_base
                ORDER BY popularity DESC
                LIMIT 5
            ''')
            
            popular_concepts = [
                {
                    'concept': row[0],
                    'popularity': row[1],
                    'last_accessed': row[2]
                } 
                for row in cursor.fetchall()
            ]
        
        stats = {
            'total_queries': total_queries,
            'total_facts': total_facts,
            'total_concepts': total_concepts,
            'recent_queries_24h': recent_queries,
            'average_confidence': round(avg_confidence, 3),
            'top_sources': top_sources,
            'popular_concepts': popular_concepts,
            'database_size_mb': round(Path(self.db_path).stat().st_size / (1024 * 1024), 2)
        }
        
        logger.info(f"Knowledge base stats: {total_queries} queries, {total_facts} facts, {total_concepts} concepts")
        return stats
    
    def clear_cache(self, older_than_days: int = 30) -> int:
        """
        Clear old cached data
        
        Args:
            older_than_days: Remove entries older than this many days
            
        Returns:
            Number of queries removed
        """
        cutoff_time = time.time() - (older_than_days * 86400)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Count queries to be removed
            cursor.execute('SELECT COUNT(*) FROM queries WHERE timestamp < ?', (cutoff_time,))
            count_to_remove = cursor.fetchone()[0]
            
            if count_to_remove > 0:
                # Delete old queries (facts will be cascade deleted)
                cursor.execute('DELETE FROM queries WHERE timestamp < ?', (cutoff_time,))
                
                # Delete orphaned concepts with low popularity
                cursor.execute('''
                    DELETE FROM knowledge_base 
                    WHERE popularity < 0.1 AND last_accessed < ?
                ''', (cutoff_time,))
                
                conn.commit()
                
                logger.info(f"Cleared {count_to_remove} old queries from cache")
            else:
                logger.info("No old queries to clear")
        
        return count_to_remove
    
    def _generate_query_hash(self, query: str, user_id: str, mode: str, submode: str) -> str:
        """
        Generate a hash for query caching
        
        Args:
            query: Query text
            user_id: User ID
            mode: Processing mode
            submode: Response sub-mode
            
        Returns:
            Hash string
        """
        hash_input = f"{query.lower().strip()}|{user_id}|{mode}|{submode}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def _calculate_similarity(self, query1: str, query2: str) -> float:
        """
        Calculate similarity between two queries
        
        Args:
            query1: First query
            query2: Second query
            
        Returns:
            Similarity score between 0 and 1
        """
        words1 = set(query1.lower().split())
        words2 = set(query2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _update_knowledge_base(self, key_information: Dict[str, Any], query_id: int):
        """
        Update knowledge base with extracted key information
        
        Args:
            key_information: Key information extracted from query
            query_id: Query ID for reference
        """
        concepts_to_update = []
        
        # Extract concepts from key information
        if 'definitions' in key_information:
            for definition in key_information['definitions']:
                term = definition.get('term', '')
                if term:
                    concepts_to_update.append((term, definition.get('definition', '')))
        
        if 'main_facts' in key_information:
            for fact in key_information['main_facts']:
                # Extract key terms from fact
                fact_words = fact.get('fact', '').split()[:3]  # First 3 words
                if fact_words:
                    concept = ' '.join(fact_words)
                    concepts_to_update.append((concept, fact.get('fact', '')))
        
        # Update knowledge base
        current_time = time.time()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for concept, definition in concepts_to_update:
                if len(concept) > 2 and len(concept) < 100:  # Reasonable length
                    cursor.execute('''
                        INSERT OR REPLACE INTO knowledge_base
                        (concept, definition, popularity, last_accessed, created_at, updated_at)
                        VALUES (?, ?, 
                            COALESCE((SELECT popularity FROM knowledge_base WHERE concept = ?) + 0.1, 1.0),
                            ?, 
                            COALESCE((SELECT created_at FROM knowledge_base WHERE concept = ?), ?),
                            ?
                        )
                    ''', (
                        concept,
                        definition,
                        concept,
                        current_time,
                        concept,
                        current_time,
                        current_time
                    ))
            
            conn.commit()
    
    def get_concept_info(self, concept: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific concept
        
        Args:
            concept: Concept name
            
        Returns:
            Concept information or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT concept, definition, popularity, last_accessed, created_at, updated_at
                FROM knowledge_base
                WHERE concept = ?
            ''', (concept,))
            
            row = cursor.fetchone()
            if row:
                # Update last accessed time
                cursor.execute('''
                    UPDATE knowledge_base
                    SET last_accessed = ?
                    WHERE concept = ?
                ''', (time.time(), concept))
                
                conn.commit()
                
                return {
                    'concept': row[0],
                    'definition': row[1],
                    'popularity': row[2],
                    'last_accessed': row[3],
                    'created_at': row[4],
                    'updated_at': row[5]
                }
        
        return None
    
    def search_knowledge_base(self, search_term: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search the knowledge base for concepts
        
        Args:
            search_term: Term to search for
            limit: Maximum number of results
            
        Returns:
            List of matching concepts
        """
        search_pattern = f"%{search_term.lower()}%"
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT concept, definition, popularity, last_accessed
                FROM knowledge_base
                WHERE LOWER(concept) LIKE ? OR LOWER(definition) LIKE ?
                ORDER BY popularity DESC, last_accessed DESC
                LIMIT ?
            ''', (search_pattern, search_pattern, limit))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'concept': row[0],
                    'definition': row[1],
                    'popularity': row[2],
                    'last_accessed': row[3]
                })
        
        return results