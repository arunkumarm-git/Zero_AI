import { useEffect, useState } from "react";
import Share from "./Share";
import Post from "./Post";
import styled from "styled-components";
import axios from "axios";

export default function Feed() {
  const [posts, setPosts] = useState([]);

  useEffect(() => {
    const fetchPosts = async () => {
      try {
        // Fetch from your FastAPI backend
        const res = await axios.get("http://localhost:8000/api/timeline/all");
        setPosts(res.data);
      } catch (err) {
        console.log("Error fetching posts:", err);
      }
    };
    fetchPosts();
  }, []);

  return (
    <FeedContainer>
      <div className="feedWrapper">
        {/* Your "Zero AI" Share Box */}
        <Share />

        {/* Render each post */}
        {posts.map((p) => (
          <Post key={p._id} post={p} />
        ))}
      </div>
    </FeedContainer>
  );
}

const FeedContainer = styled.div`
  flex: 5.5;
  .feedWrapper {
    padding: 20px;
  }
`;