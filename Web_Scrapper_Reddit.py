import csv
import praw
from psaw import PushshiftAPI

api = PushshiftAPI()

## Function to search for indicated amount of WSB submissions on reddit
## returns a list of titles, a list of urls, and a list of authors
def Get_Submissions_rWSB(size):
    title_list = []
    url_list = []
    author_list = []
    submissions = api.search_submissions(
                                subreddit='wallstreetbets',
                                filter=['title','url','author'],
                                limit= size
                                )
    for submission in submissions:
        title= submission.title
        url = submission.url
        author = submission.author
        
        title_list.append(title)
        url_list.append(url)
        author_list.append(author)

    return title_list, url_list, author_list

## Function to search for indicated amount of r/investing submissions on reddit
## returns a list of titles, a list of urls, and a list of authors
def Get_Submissions_rInvesting(size):
    title_list = []
    url_list = []
    author_list = []
    submissions = api.search_submissions(
                                subreddit='investing',
                                filter=['title','url','author'],
                                limit = size
                                )
    for submission in submissions:
        title= submission.title
        url = submission.url
        author = submission.author
        
        title_list.append(title)
        url_list.append(url)
        author_list.append(author)

    return title_list, url_list, author_list

## Function to search for indicated amount of submissions on indicated subreddit
## returns a list of titles, a list of urls, and a list of authors
def Get_Submissions_Any(size, subreddit):
    title_list = []
    url_list = []
    author_list = []
    submissions = api.search_submissions(
                                subreddit = subreddit,
                                filter = ['title','url','author'],
                                limit = size
                                )
    for submission in submissions:
        title= submission.title
        url = submission.url
        author = submission.author
        
        title_list.append(title)
        url_list.append(url)
        author_list.append(author)
    return title_list, url_list, author_list

## Function to search for comments given a submission URL
## returns list of authors and list of Comment body
def Get_Comments_From_Submissions(submission_url):
    author_list = []
    body_list = []
    comments = api.search_comments(url=submission_url)
    
    for comment in comments:
        author = comment.author
        body = comment
        
        author_list.append(author)
        body_list.append(comment)
    return author_list, body_list
