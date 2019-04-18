import csv



class Credbank():



    def __init__(self, tweet_file, rating_file):

        self.tweet_file = tweet_file
        self.rating_file = rating_file

        self.data_dict = self.parse_data()

    def parse_data(self):

        '''
        Returns dict
        key =  topic_key
        val = ([list of tweet_ids], class_label)
        '''

        tweet_count = 0
        with open(self.tweet_file) as myfile:

            csv_reader = csv.DictReader(myfile)
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    print(row)
                    print(f'Column names are {", ".join(row)}')
                    line_count += 1
                line_count += 1
            print(f'Processed {line_count} lines.')

            '''for line in myfile:
                if tweet_count == -1:
                    #First line useless
                    pass

                else:
                    print(line[1:100])


                tweet_count += 1


                if tweet_count == 2:
                    break'''


        print(tweet_count)





Credbank('../../Credbank/cred_event_SearchTweets_smallSample.data', '../../Credbank/cred_event_TurkRatings_smallSample.data')

