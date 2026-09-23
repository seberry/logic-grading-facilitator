from fastapi.testclient import TestClient

from app.server import app


def test_cross_origin_pages_cannot_read_or_write_grader() -> None:
    client = TestClient(app)
    read = client.get("/openapi.json", headers={"Origin": "https://unrelated.example"})
    assert read.status_code == 200
    assert "access-control-allow-origin" not in read.headers

    write = client.post("/not-a-real-route", headers={"Origin": "https://unrelated.example"})
    assert write.status_code == 403
    malformed = client.post("/not-a-real-route", headers={"Origin": "http://[broken"})
    assert malformed.status_code == 403


def test_same_origin_and_nonbrowser_writes_reach_routes() -> None:
    client = TestClient(app)
    for headers in ({"Origin": "http://testserver"}, {}):
        response = client.post("/not-a-real-route", headers=headers)
        assert response.status_code == 404


def test_cross_site_fetch_header_denies_write_without_origin() -> None:
    client = TestClient(app)
    response = client.post("/not-a-real-route", headers={"Sec-Fetch-Site": "cross-site"})
    assert response.status_code == 403
