package com.example.capstoneassistant

import android.content.Intent
import android.graphics.Color
import android.os.Bundle
import android.util.Log
import android.view.Gravity
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.drawerlayout.widget.DrawerLayout
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.google.firebase.messaging.FirebaseMessaging

data class ChatMessage(
    val text: String,
    val isUser: Boolean
)

class MainActivity : AppCompatActivity() {

    private val messages = mutableListOf<ChatMessage>()
    private lateinit var adapter: MessageAdapter

    private var currentUserName: String = "사용자"
    private var currentThemeMode: String = "dark"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        currentUserName = intent.getStringExtra("userName") ?: "사용자"
        currentThemeMode = intent.getStringExtra("themeMode") ?: "dark"

        FirebaseMessaging.getInstance().token
            .addOnCompleteListener { task ->
                if (!task.isSuccessful) {
                    Log.w("FCM_TOKEN", "FCM 토큰 가져오기 실패", task.exception)
                    return@addOnCompleteListener
                }

                val token = task.result
                Log.d("FCM_TOKEN", token)
            }

        val drawerLayout = findViewById<DrawerLayout>(R.id.drawerLayout)

        val mainRootLayout = findViewById<LinearLayout>(R.id.mainRootLayout)
        val topBarLayout = findViewById<LinearLayout>(R.id.topBarLayout)
        val inputLayout = findViewById<LinearLayout>(R.id.inputLayout)
        val menuLayout = findViewById<LinearLayout>(R.id.menuLayout)

        val chatRecyclerView = findViewById<RecyclerView>(R.id.chatRecyclerView)
        val messageEditText = findViewById<EditText>(R.id.messageEditText)
        val sendButton = findViewById<Button>(R.id.sendButton)

        val menuButton = findViewById<TextView>(R.id.menuButton)
        val titleText = findViewById<TextView>(R.id.titleText)
        val menuTitleText = findViewById<TextView>(R.id.menuTitleText)
        val menuDivider = findViewById<View>(R.id.menuDivider)

        val scheduleMenu = findViewById<TextView>(R.id.scheduleMenu)
        val routeMenu = findViewById<TextView>(R.id.routeMenu)
        val clothesMenu = findViewById<TextView>(R.id.clothesMenu)
        val alarmMenu = findViewById<TextView>(R.id.alarmMenu)
        val weatherMenu = findViewById<TextView>(R.id.weatherMenu)
        val settingMenu = findViewById<TextView>(R.id.settingMenu)

        applyTheme(
            currentThemeMode,
            drawerLayout,
            mainRootLayout,
            topBarLayout,
            inputLayout,
            menuLayout,
            chatRecyclerView,
            messageEditText,
            menuButton,
            titleText,
            menuTitleText,
            menuDivider,
            listOf(
                scheduleMenu,
                routeMenu,
                clothesMenu,
                alarmMenu,
                weatherMenu,
                settingMenu
            )
        )

        adapter = MessageAdapter(messages)

        chatRecyclerView.adapter = adapter
        chatRecyclerView.layoutManager = LinearLayoutManager(this)

        addMessage(
            "안녕하세요, ${currentUserName}님! 오늘 일정을 확인해 드릴까요?",
            false,
            chatRecyclerView
        )

        menuButton.setOnClickListener {
            drawerLayout.openDrawer(Gravity.START)
        }

        scheduleMenu.setOnClickListener {
            drawerLayout.closeDrawer(Gravity.START)

            addMessage(
                "${currentUserName}님, 일정 관리 기능입니다. 등록된 일정을 확인하거나 새 일정을 추가할 수 있습니다.",
                false,
                chatRecyclerView
            )
        }

        routeMenu.setOnClickListener {
            drawerLayout.closeDrawer(Gravity.START)

            val intent = Intent(this, RouteActivity::class.java)
            startActivity(intent)
        }

        clothesMenu.setOnClickListener {
            drawerLayout.closeDrawer(Gravity.START)

            val intent = Intent(this, ClothesActivity::class.java)
            startActivity(intent)
        }

        alarmMenu.setOnClickListener {
            drawerLayout.closeDrawer(Gravity.START)

            addMessage(
                "${currentUserName}님, 알람 기능입니다. 원하는 시간에 알림을 받을 수 있도록 설정할 수 있습니다.",
                false,
                chatRecyclerView
            )
        }

        weatherMenu.setOnClickListener {
            drawerLayout.closeDrawer(Gravity.START)

            addMessage(
                "${currentUserName}님, 날씨 기능입니다. 현재 날씨를 확인하고 날씨에 맞는 추천을 받을 수 있습니다.",
                false,
                chatRecyclerView
            )
        }

        settingMenu.setOnClickListener {
            drawerLayout.closeDrawer(Gravity.START)

            addMessage(
                "${currentUserName}님, 설정 화면은 추후 추가 예정입니다.",
                false,
                chatRecyclerView
            )
        }

        sendButton.setOnClickListener {
            val userMessage = messageEditText.text.toString()

            if (userMessage.isNotBlank()) {
                addMessage(userMessage, true, chatRecyclerView)

                val aiReply = getAiResponse(userMessage)

                addMessage(aiReply, false, chatRecyclerView)

                messageEditText.text.clear()
            }
        }
    }

    private fun applyTheme(
        themeMode: String,
        drawerLayout: DrawerLayout,
        mainRootLayout: LinearLayout,
        topBarLayout: LinearLayout,
        inputLayout: LinearLayout,
        menuLayout: LinearLayout,
        chatRecyclerView: RecyclerView,
        messageEditText: EditText,
        menuButton: TextView,
        titleText: TextView,
        menuTitleText: TextView,
        menuDivider: View,
        menuTexts: List<TextView>
    ) {
        if (themeMode == "light") {
            val backgroundColor = Color.parseColor("#F7F7F7")
            val barColor = Color.parseColor("#FFFFFF")
            val inputColor = Color.parseColor("#EEEEEE")
            val textColor = Color.parseColor("#222222")
            val hintColor = Color.parseColor("#777777")
            val dividerColor = Color.parseColor("#DDDDDD")

            drawerLayout.setBackgroundColor(backgroundColor)
            mainRootLayout.setBackgroundColor(backgroundColor)
            chatRecyclerView.setBackgroundColor(backgroundColor)
            topBarLayout.setBackgroundColor(barColor)
            inputLayout.setBackgroundColor(barColor)
            menuLayout.setBackgroundColor(barColor)
            messageEditText.setBackgroundColor(inputColor)

            menuButton.setTextColor(textColor)
            titleText.setTextColor(textColor)
            menuTitleText.setTextColor(textColor)
            messageEditText.setTextColor(textColor)
            messageEditText.setHintTextColor(hintColor)
            menuDivider.setBackgroundColor(dividerColor)

            menuTexts.forEach {
                it.setTextColor(textColor)
            }
        } else {
            val backgroundColor = Color.parseColor("#121212")
            val barColor = Color.parseColor("#1E1E1E")
            val inputColor = Color.parseColor("#2A2A2A")
            val textColor = Color.parseColor("#FFFFFF")
            val hintColor = Color.parseColor("#AAAAAA")
            val dividerColor = Color.parseColor("#444444")

            drawerLayout.setBackgroundColor(backgroundColor)
            mainRootLayout.setBackgroundColor(backgroundColor)
            chatRecyclerView.setBackgroundColor(backgroundColor)
            topBarLayout.setBackgroundColor(barColor)
            inputLayout.setBackgroundColor(barColor)
            menuLayout.setBackgroundColor(barColor)
            messageEditText.setBackgroundColor(inputColor)

            menuButton.setTextColor(textColor)
            titleText.setTextColor(textColor)
            menuTitleText.setTextColor(textColor)
            messageEditText.setTextColor(textColor)
            messageEditText.setHintTextColor(hintColor)
            menuDivider.setBackgroundColor(dividerColor)

            menuTexts.forEach {
                it.setTextColor(textColor)
            }
        }
    }

    private fun addMessage(
        text: String,
        isUser: Boolean,
        recyclerView: RecyclerView
    ) {
        messages.add(ChatMessage(text, isUser))
        adapter.notifyItemInserted(messages.size - 1)
        recyclerView.scrollToPosition(messages.size - 1)
    }

    private fun getAiResponse(userMessage: String): String {
        return when {
            userMessage.contains("옷") ||
                    userMessage.contains("옷장") ||
                    userMessage.contains("코디") ->
                "${currentUserName}님, 옷 추천 메뉴에서 날씨에 맞는 옷을 확인할 수 있습니다."

            userMessage.contains("날씨") ->
                "${currentUserName}님, 날씨 메뉴에서 현재 날씨 정보를 확인할 수 있습니다."

            userMessage.contains("일정") ->
                "${currentUserName}님, 등록된 일정을 확인해드릴게요."

            userMessage.contains("알람") ->
                "${currentUserName}님, 알람 기능에서 원하는 시간에 알림을 설정할 수 있습니다."

            userMessage.contains("길") ||
                    userMessage.contains("경로") ->
                "${currentUserName}님, 경로 찾기 메뉴에서 출발지와 목적지를 입력해주세요."

            else ->
                "${currentUserName}님, 아직 실제 AI 백엔드 연결 전입니다. 백엔드 연결 후 실제 답변이 표시됩니다."
        }
    }
}

class MessageAdapter(
    private val messages: List<ChatMessage>
) : RecyclerView.Adapter<MessageAdapter.MessageViewHolder>() {

    class MessageViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        val messageContainer: LinearLayout =
            itemView.findViewById(R.id.messageContainer)

        val messageText: TextView =
            itemView.findViewById(R.id.messageText)
    }

    override fun onCreateViewHolder(
        parent: ViewGroup,
        viewType: Int
    ): MessageViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_message, parent, false)

        return MessageViewHolder(view)
    }

    override fun onBindViewHolder(
        holder: MessageViewHolder,
        position: Int
    ) {
        val message = messages[position]

        holder.messageText.text = message.text

        val layoutParams =
            holder.messageText.layoutParams as LinearLayout.LayoutParams

        if (message.isUser) {
            holder.messageContainer.gravity = Gravity.END
            holder.messageText.setBackgroundResource(R.drawable.bubble_user)
            layoutParams.marginStart = 80
            layoutParams.marginEnd = 8
        } else {
            holder.messageContainer.gravity = Gravity.START
            holder.messageText.setBackgroundResource(R.drawable.bubble_ai)
            layoutParams.marginStart = 8
            layoutParams.marginEnd = 80
        }

        holder.messageText.layoutParams = layoutParams
    }

    override fun getItemCount(): Int {
        return messages.size
    }
}