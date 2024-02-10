import asyncio
from typing import Any, Generator, List, cast, Dict, Union

import pytest
from llama_index.schema import (
    NodeRelationship,
    RelatedNodeInfo,
    TextNode,
)
from llama_iris import IRISVectorStore
from llama_index.vector_stores.loading import load_vector_store
from llama_index.vector_stores.types import (
    VectorStoreQuery,
)

import intersystems_iris.dbapi._DBAPI as dbapi

TEST_TABLE_NAME = "lorem_ipsum"
TEST_SCHEMA_NAME = "test"
TEST_EMBED_DIM = 2

iris_not_available = False


@pytest.fixture(scope="session")
def connection_string(request):
    return request.config.getoption("--dburi")


def _get_sample_vector(num: float) -> List[float]:
    """
    Get sample embedding vector of the form [num, 1, 1, ..., 1]
    where the length of the vector is TEST_EMBED_DIM.
    """
    return [num] + [1.0] * (TEST_EMBED_DIM - 1)


@pytest.fixture(scope="session")
def conn(connection_string) -> Any:
    from sqlalchemy import make_url

    url = make_url(connection_string)
    PARAMS: Dict[str, Union[str, int]] = {
        "hostname": url.host,
        "username": url.username,
        "password": url.password,
        "port": url.port,
        "namespace": url.database,
    }

    return dbapi.connect(**PARAMS)  # type: ignore


@pytest.fixture()
def db(conn: Any) -> Generator:
    conn.autocommit = True

    with conn.cursor() as c:
        c.execute(f"CREATE SCHEMA IF NOT EXISTS  {TEST_SCHEMA_NAME}")
        conn.commit()
    yield
    with conn.cursor() as c:
        c.execute(f"DROP SCHEMA IF EXISTS {TEST_SCHEMA_NAME} CASCADE")
        conn.commit()


@pytest.fixture()
def iris(db: None, connection_string) -> Any:
    iris = IRISVectorStore.from_params(
        connection_string=connection_string,
        table_name=TEST_TABLE_NAME,
        schema_name=TEST_SCHEMA_NAME,
        embed_dim=TEST_EMBED_DIM,
    )

    yield iris

    asyncio.run(iris.close())


@pytest.fixture(scope="session")
def node_embeddings() -> List[TextNode]:
    return [
        TextNode(
            text="lorem ipsum",
            id_="aaa",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="aaa")},
            embedding=_get_sample_vector(1.0),
        ),
        TextNode(
            text="dolor sit amet",
            id_="bbb",
            relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id="bbb")},
            extra_info={"test_key": "test_value"},
            embedding=_get_sample_vector(0.1),
        ),
    ]


@pytest.mark.skipif(iris_not_available, reason="IRIS db is not available")
@pytest.mark.asyncio()
async def test_add_to_db_and_query(
    iris: IRISVectorStore,
    node_embeddings: List[TextNode],
) -> None:
    iris.add(node_embeddings)
    assert isinstance(iris, IRISVectorStore)
    assert hasattr(iris, "_engine")
    q = VectorStoreQuery(query_embedding=_get_sample_vector(1.0), similarity_top_k=1)
    res = iris.query(q)
    assert res.nodes
    assert len(res.nodes) == 1
    assert res.nodes[0].node_id == "aaa"


@pytest.mark.skipif(iris_not_available, reason="IRIS db is not available")
@pytest.mark.asyncio()
async def test_add_to_db_delete_and_query(
    iris: IRISVectorStore,
    node_embeddings: List[TextNode],
) -> None:
    iris.add(node_embeddings)
    assert isinstance(iris, IRISVectorStore)
    assert hasattr(iris, "_engine")
    q = VectorStoreQuery(query_embedding=_get_sample_vector(1.0), similarity_top_k=1)
    iris.delete("aaa")
    res = iris.query(q)
    assert res.nodes
    assert len(res.nodes) == 1
    assert res.nodes[0].node_id == "bbb"  # aaa deleted and bbb left


@pytest.mark.skipif(iris_not_available, reason="IRIS db is not available")
@pytest.mark.asyncio()
async def test_save_load(
    iris: IRISVectorStore, node_embeddings: List[TextNode]
) -> None:
    iris.add(node_embeddings)
    assert isinstance(iris, IRISVectorStore)
    assert hasattr(iris, "_engine")

    q = VectorStoreQuery(query_embedding=_get_sample_vector(0.1), similarity_top_k=1)

    res = iris.query(q)
    assert res.nodes
    assert len(res.nodes) == 1
    assert res.nodes[0].node_id == "bbb"

    iris_dict = iris.to_dict()
    await iris.close()

    loaded_pg = cast(IRISVectorStore, load_vector_store(iris_dict))
    assert not hasattr(loaded_pg, "_engine")
    loaded_pg_dict = loaded_pg.to_dict()
    for key, val in iris.to_dict().items():
        assert loaded_pg_dict[key] == val

    res = loaded_pg.query(q)
    assert hasattr(loaded_pg, "_engine")
    assert res.nodes
    assert len(res.nodes) == 1
    assert res.nodes[0].node_id == "bbb"

    await loaded_pg.close()
