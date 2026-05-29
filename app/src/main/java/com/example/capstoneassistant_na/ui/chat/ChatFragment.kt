package com.example.capstoneassistant_na.ui.chat

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.EditText
import android.widget.ImageButton
import android.widget.ScrollView
import android.widget.TextView
import androidx.fragment.app.Fragment
import com.example.capstoneassistant_na.R
import com.example.capstoneassistant_na.data.api.RetrofitClient
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class ChatFragment : Fragment() {

    private lateinit var chatLog: TextView
    private lateinit var inputField: EditText
    private lateinit var sendButton: ImageButton
    private lateinit var scrollView: ScrollView

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        return inflater.inflate(R.layout.fragment_chat, container, false)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        chatLog = view.findViewById(R.id.chatLog)
        inputField = view.findViewById(R.id.inputField)
        sendButton = view.findViewById(R.id.sendButton)
        scrollView = view.findViewById(R.id.scrollView)

        appendMessage("어시스턴트", "안녕하세요! 일정, 옷 추천, 경로 등 무엇이든 물어보세요 😊")

        sendButton.setOnClickListener {
            val text = inputField.text.toString().trim()
            if (text.isNotEmpty()) {
                inputField.setText("")
                sendMessage(text)
            }
        }
    }

    private fun sendMessage(message: String) {
        appendMessage("나", message)

        CoroutineScope(Dispatchers.IO).launch {
            try {
                val response = RetrofitClient.instance.sendMessage(
                    mapOf("message" to message, "session_id" to "default")
                )
                val reply = response["reply"] ?: "응답을 받지 못했어요."
                withContext(Dispatchers.Main) {
                    appendMessage("어시스턴트", reply)
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    appendMessage("어시스턴트", "서버 연결에 실패했어요. 인터넷을 확인해주세요.")
                }
            }
        }
    }

    private fun appendMessage(sender: String, message: String) {
        val current = chatLog.text.toString()
        val newText = if (current.isEmpty()) "[$sender]\n$message"
                      else "$current\n\n[$sender]\n$message"
        chatLog.text = newText
        scrollView.post { scrollView.fullScroll(ScrollView.FOCUS_DOWN) }
    }
}
