from bs4 import BeautifulSoup
import os
import requests


class CQRI():

    def __init__(self, twitter_path):

        '''

        :param twitter_path: relative path to the twitter.txt file
        '''

        self.twitter_path = twitter_path

        self.tweet_dict = self.parse()


    def parse(self):

        '''
        Returns dict with
        key : Event ID
        val : ([list of tweets ID], label)
        '''


        with open(self.twitter_path) as myfile:

            tweet_dict = {}

            for line in myfile:

                parts = line.rstrip().split(' ')

                #event id
                eid = parts[0].split(':')[1]

                #label
                label = int(parts[1].split(':')[1])

                #tweets
                tweet_ids = [tid for tid in parts[2:]]

                tweet_dict[eid] = (tweet_ids, label)

                #print('building entry', eid, label, tweet_ids)

            return tweet_dict


    def get_dict(self):
        return self.tweet_dict


    def download_tweets(self, output_dir):

        # Create directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for key, val in self.tweet_dict.items():

            tweet_ids = val[0]

            print('Getting tweets of', key)

            for tweet_id in tweet_ids:

                filename = output_dir + '/' + str(tweet_id) + '.txt'

                tweet_url = 'https://twitter.com/statuses/'+tweet_id

                r = requests.get(tweet_url)
                soup = BeautifulSoup(r.text, 'html.parser')

                tweets = soup.findAll('p', class_='tweet-text')

                if len(tweets) > 0:
                    with open(filename, 'w', encoding='utf-8') as myfile:
                        myfile.write(tweets[0].text)
                        myfile.close()
                else:
                    print('Warning: tweet did not exist')



            break




dataset = CQRI('../Twitter.txt')
dataset.download_tweets('tweets')