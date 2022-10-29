import csv
import praw
from psaw import PushshiftAPI

api = PushshiftAPI()

## Function to search for indicated amount of WSB submissions on reddit
## returns a list of titles, a list of urls, and a list of authors
def Get_Submissions_rWSB(size):
    submission_list = []
    submissions = api.search_submissions(
                                subreddit='wallstreetbets',
                                filter=['title','url','author'],
                                limit= size
                                )
    for submission in submissions:
        title= submission.title
        url = submission.url
        author = submission.author

        submission_list.append((title,url,author))
    return submission_list

## Function to search for indicated amount of r/investing submissions on reddit
## returns a list of titles, a list of urls, and a list of authors
def Get_Submissions_rInvesting(size):
    submission_list = []
    submissions = api.search_submissions(
                                subreddit='investing',
                                filter=['title','url','author'],
                                limit = size
                                )
    for submission in submissions:
        title= submission.title
        url = submission.url
        author = submission.author

        submission_list.append((title,url,author))
    return submission_list

## Function to search for indicated amount of submissions on indicated subreddit
## returns a list of titles, a list of urls, and a list of authors
def Get_Submissions_Any(size, subreddit):
    submission_list = []
    submissions = api.search_submissions(
                                subreddit = subreddit,
                                filter = ['title','url','author'],
                                limit = size
                                )
    for submission in submissions:
        title= submission.title
        url = submission.url
        author = submission.author

        submission_list.append((title,url,author))
    return submission_list

## Function to search for comments given a submission URL
## returns list of authors and list of Comment body
def Get_Comments_From_Submissions(submission_url):
    comment_list = []
    comments = api.search_comments(url=submission_url)

    for comment in comments:
        author = comment.author
        body = comment

        comment_list.append((author,body))
    return comment_list
