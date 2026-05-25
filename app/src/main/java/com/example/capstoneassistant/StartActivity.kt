package com.example.capstoneassistant

import android.content.Intent
import android.os.Bundle
import android.widget.Button
import android.widget.RadioButton
import android.widget.RadioGroup
import androidx.appcompat.app.AppCompatActivity

class StartActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_start)

        val userRadioGroup = findViewById<RadioGroup>(R.id.userRadioGroup)
        val themeRadioGroup = findViewById<RadioGroup>(R.id.themeRadioGroup)
        val startButton = findViewById<Button>(R.id.startButton)

        startButton.setOnClickListener {
            val selectedUserId = userRadioGroup.checkedRadioButtonId
            val selectedThemeId = themeRadioGroup.checkedRadioButtonId

            val selectedUserButton = findViewById<RadioButton>(selectedUserId)
            val userName = selectedUserButton.text.toString()

            val themeMode =
                if (selectedThemeId == R.id.lightModeButton) {
                    "light"
                } else {
                    "dark"
                }

            val intent = Intent(this, MainActivity::class.java)
            intent.putExtra("userName", userName)
            intent.putExtra("themeMode", themeMode)
            startActivity(intent)
        }
    }
}