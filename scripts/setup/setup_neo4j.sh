#!/usr/bin/env bash
set -euo pipefail

NEO4J_URI="${NEO4J_URI:-bolt://localhost:7687}"
NEO4J_USER="${NEO4J_USER:-neo4j}"
NEO4J_PASSWORD="${NEO4J_PASSWORD:-change-me}"
NEO4J_HTTP="${NEO4J_HTTP:-http://localhost:7474}"
MAX_WAIT=120

echo "Waiting for Neo4j to be ready at ${NEO4J_HTTP}..."
ELAPSED=0
until wget -qO- "${NEO4J_HTTP}" > /dev/null 2>&1; do
    if [ $ELAPSED -ge $MAX_WAIT ]; then
        echo "ERROR: Neo4j did not start within ${MAX_WAIT}s"
        exit 1
    fi
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    echo "  Waiting... (${ELAPSED}s)"
done
echo "Neo4j is ready."

run_cypher() {
    cypher-shell -a "${NEO4J_URI}" -u "${NEO4J_USER}" -p "${NEO4J_PASSWORD}" "$@"
}

echo "Creating constraints and indexes..."

run_cypher "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE;"
run_cypher "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE;"
run_cypher "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name);"
run_cypher "CREATE INDEX document_collection IF NOT EXISTS FOR (d:Document) ON (d.collection);"
run_cypher "CREATE INDEX document_date IF NOT EXISTS FOR (d:Document) ON (d.date);"

echo "Creating fulltext index for entity search..."
run_cypher "
CREATE FULLTEXT INDEX entityFullText IF NOT EXISTS
FOR (e:Entity)
ON EACH [e.name, e.description, e.aliases];
"

echo "Neo4j setup complete."
