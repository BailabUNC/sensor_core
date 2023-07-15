from sqlitedict import SqliteDict


def _save_channel(key, value, cache_file="cache.sqlite3"):
    try:
        with SqliteDict(cache_file) as mydict:
            mydict[key] = value  # Using dict[key] to store
            mydict.commit()  # Need to commit() to actually flush the data
    except Exception as ex:
        print("Error during storing data (Possibly unsupported):", ex)


def _load_channel(key, cache_file="cache.sqlite3"):
    try:
        with SqliteDict(cache_file) as mydict:
            value = mydict[key]
        return value
    except Exception as ex:
        print("Error during loading data:", ex)


def save_channels(key, value, cache_file="cache.sqlite3"):
    try:
        data = _load_channel(key, cache_file)
        data.append(value)
        _save_channel(key, data, cache_file)
    except Exception as ex:
        print("Error during saving data", ex)

