



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


