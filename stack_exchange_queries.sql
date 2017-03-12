-- Query for open questions and user metrics
SELECT TOP 40000
  PostsWithDeleted.Id AS PostId, 
  PostsWithDeleted.Score, 
  PostsWithDeleted.ViewCount, 
  PostsWithDeleted.Body, 
  PostsWithDeleted.OwnerUserId, 
  PostsWithDeleted.Title,
  PostsWithDeleted.Tags,
  PostsWithDeleted.AnswerCount,
  PostsWithDeleted.CommentCount,
  PostsWithDeleted.FavoriteCount,
  PostsWithDeleted.AcceptedAnswerId,
  PostsWithDeleted.CreationDate,
  PostsWithDeleted.ClosedDate,
  PostsWithDeleted.DeletionDate,
  Users.Reputation AS OwnerReputation,
  Users.UpVotes AS OwnerUpVoteCount
FROM PostsWithDeleted 
  join Users on PostsWithDeleted.OwnerUserId = Users.Id
WHERE  
  PostsWithDeleted.AcceptedAnswerId IS NOT NULL
; 


-- Query for closed questions and user metrics
select top 1000
  PostsWithDeleted.Id as PostId, 
  PostsWithDeleted.Score, 
  PostsWithDeleted.ViewCount, 
  PostsWithDeleted.Body, 
  PostsWithDeleted.OwnerUserId, 
  PostsWithDeleted.Title,
  PostsWithDeleted.Tags,
  PostsWithDeleted.AnswerCount,
  PostsWithDeleted.CommentCount,
  PostsWithDeleted.CreationDate,
  PostsWithDeleted.DeletionDate,
  PostsWithDeleted.ClosedDate,
  Users.Reputation,
  Users.UpVotes,
  Badges.Class,
  PostHistory.Id,
  PostHistory.Comment,
  CloseReasonTypes.Name,
  CloseReasonTypes.Description
from PostsWithDeleted 
  join Users on PostsWithDeleted.OwnerUserId = Users.Id
  join PostHistory on PostHistory.PostId = PostsWithDeleted.Id
  join CloseReasonTypes on PostHistory.Comment = CloseReasonTypes.Id
  join Badges on Users.Id = Badges.UserId
  where PostHistory.PostHistoryTypeId =10 and (PostHistory.Comment = 101 or PostHistory.Comment = 102 or PostHistory.Comment = 103 or PostHistory.Comment = 104 or PostHistory.Comment = 105);