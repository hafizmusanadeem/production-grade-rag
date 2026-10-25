def scroll_all(client, collection, with_payload=True):
    offset=None
    while True:
        points,offset=client.scroll(collection_name=collection,limit=256,offset=offset,
                                   with_payload=with_payload,with_vectors=False)
        yield from points
        if offset is None: break
