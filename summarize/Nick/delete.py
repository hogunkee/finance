from pymongo import MongoClient

client = MongoClient('localhost', 27017)
db = client['reddit']

#db.submission.find()
remove_field ={ 
    "likes" : 1, 
    "selftext_html" : 1, 
    "banned_by" : 1, 
    "media_embed" : 1,
    "id" : 1, 
    "link_flair_css_class" : 1, 
    "over_18" : 1, 
    "clicked" : 1, 
    "url" : 1,
    "edited" : 1, 
    "name" : 1,
    "selftext" : 1, 
    "approved_by" : 1, 
    "hidden" : 1, 
    "author_flair_css_class" : 1, 
    "media" : 1, 
    "saved" : 1, 
    "author" : 1,
    "created_utc" : 1,
    "thumbnail" : 1,
    "num_reports" : 1, 
    "distinguished" : 1, 
    "link_flair_text" : 1, 
    "subreddit_id" : 1,
    "author_flair_text" : 1, 
    "permalink" : 1
}
db.submission.update({}, {'$unset': remove_field} , {'multi': true})
