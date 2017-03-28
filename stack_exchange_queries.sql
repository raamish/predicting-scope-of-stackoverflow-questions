-- OPEN QUESTIONS
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


-- CLOSED QUESTIONS
SELECT TOP 60000
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
  Users.Reputation as OwnerReputation,
  Users.UpVotes as OwnerUpVoteCount,
  PostHistory.Id,
  PostHistory.Comment,
  CloseReasonTypes.Name,
  CloseReasonTypes.Description
FROM PostsWithDeleted 
  JOIN Users on PostsWithDeleted.OwnerUserId = Users.Id
  JOIN PostHistory on PostHistory.PostId = PostsWithDeleted.Id
  JOIN CloseReasonTypes on PostHistory.Comment = CloseReasonTypes.Id
  WHERE PostHistory.PostHistoryTypeId =10 AND (PostHistory.Comment = 102 OR PostHistory.Comment = 103 OR PostHistory.Comment = 104 OR PostHistory.Comment = 105)
;