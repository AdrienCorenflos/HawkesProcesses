from twython import Twython
import datetime as dt
import numpy as np
from .univariate import HawkesProcess


def get_tweets_arrival_process(app_key, app_key_secret, access_token, access_token_secret, query):
    t = Twython(app_key=app_key,
                app_secret=app_key_secret,
                oauth_token=access_token,
                oauth_token_secret=access_token_secret)

    search = t.search(q=query,
                      count=1000)
    tic = dt.datetime.now(dt.timezone.utc)
    tweets = search['statuses']
    arrival_times = [tweet['created_at'] for tweet in tweets]
    times = [dt.datetime.strptime(time, "%a %b %d %H:%M:%S %z %Y") for time in arrival_times]
    times = list(reversed(times))
    T = (tic - times[0]).total_seconds()
    tweet_times = np.array([time.total_seconds() / T for time in np.array(times) - times[0]])
    tweet_hawkes = HawkesProcess.fit(tweet_times)
    return tweet_hawkes
