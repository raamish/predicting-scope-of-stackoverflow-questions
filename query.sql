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