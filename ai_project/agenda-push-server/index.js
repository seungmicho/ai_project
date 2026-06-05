const express = require("express");
const cors = require("cors");
const admin = require("firebase-admin");
const axios = require("axios");
const cron = require("node-cron");

const serviceAccount = require("./serviceAccountKey.json");

admin.initializeApp({
  credential: admin.credential.cert(serviceAccount),
});

const app = express();

app.use(cors());
app.use(express.json());

const FCM_TOKEN = "dtRjV5U0Rb6lji2cgIjXCQ:APA91bFSteLzZnMkaBtpS2MN6Yk2K1Ivmf8uXEVpJnhAbUwhnAJHsI1cTjw96p7lejvuJ2k-eZ2VDq9dFpYls0dh_kDj8rft_tzwCKg9IJhR78ewL1Sj6vM";

const sentEvents = new Set();

async function sendPush(token, title, body) {
  try {
    const messageId = await admin.messaging().send({
      token: token,
      notification: {
        title,
        body,
      },
    });
    console.log("Push sent:", messageId);
  } catch (error) {
    console.error("Push error:", error);
  }
}

/*
  수동 푸시 테스트 API
*/
app.post("/send-notification", async (req, res) => {
  const { token, title, body } = req.body;
  try {
    const messageId = await admin.messaging().send({
      token: token,
      notification: {
        title: title,
        body: body,
      },
    });
    res.status(200).send("Push notification sent: " + messageId);
  } catch (error) {
    console.error("FCM Error:", error);
    res.status(500).send(error.message);
  }
});

/*
  1분마다 Google Calendar 확인
*/
cron.schedule("* * * * *", async () => {
  console.log("Checking calendar...");

  try {
    const response = await axios.get("http://127.0.0.1:8000/schedules/today");
    const events = response.data.data || [];

    console.log(events);

    const now = Date.now();

    for (const event of events) {
      const startTime = new Date(event.start).getTime();
      const diffMinutes = Math.round((startTime - now) / 60000);

      console.log(`${event.title} : ${diffMinutes} minutes left`);

      if (
        diffMinutes <= 10 &&
        diffMinutes >= 9 &&
        !sentEvents.has(event.id)
      ) {
        await sendPush(FCM_TOKEN, "일정 알림", `${event.title} 일정이 10분 후 시작됩니다.`);
        sentEvents.add(event.id);
        console.log(`Notification sent for ${event.title}`);
      }
    }
  } catch (error) {
    console.error("Calendar error:", error.message);
  }
});

app.listen(3000, () => {
  console.log("Push server running on port 3000");
});