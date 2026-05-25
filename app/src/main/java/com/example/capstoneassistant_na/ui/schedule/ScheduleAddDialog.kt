package com.example.capstoneassistant_na.ui.schedule

import android.app.DatePickerDialog
import android.app.Dialog
import android.app.TimePickerDialog
import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import androidx.fragment.app.DialogFragment
import com.example.capstoneassistant_na.R
import java.time.LocalDateTime
import java.util.Calendar
import android.view.ViewGroup

class ScheduleAddDialog(
    private val onConfirm: (title: String, description: String, date: String) -> Unit
) : DialogFragment() {

    private var selectedDateTime: LocalDateTime = LocalDateTime.now()

    override fun onCreateDialog(savedInstanceState: Bundle?): Dialog {
        val dialog = Dialog(requireContext())
        dialog.setContentView(R.layout.dialog_schedule_add)

        val etTitle = dialog.findViewById<EditText>(R.id.etTitle)
        val etDescription = dialog.findViewById<EditText>(R.id.etDescription)
        val tvDateTime = dialog.findViewById<TextView>(R.id.tvDateTime)
        val btnPickDate = dialog.findViewById<Button>(R.id.btnPickDate)
        val btnConfirm = dialog.findViewById<Button>(R.id.btnConfirm)
        val btnCancel = dialog.findViewById<Button>(R.id.btnCancel)

        btnPickDate.setOnClickListener {
            val cal = Calendar.getInstance()
            DatePickerDialog(requireContext(), { _, y, m, d ->
                TimePickerDialog(requireContext(), { _, h, min ->
                    selectedDateTime = LocalDateTime.of(y, m + 1, d, h, min)
                    tvDateTime.text = "${y}-${m+1}-${d} ${h}:${min.toString().padStart(2,'0')}"
                }, cal.get(Calendar.HOUR_OF_DAY), cal.get(Calendar.MINUTE), true).show()
            }, cal.get(Calendar.YEAR), cal.get(Calendar.MONTH), cal.get(Calendar.DAY_OF_MONTH)).show()
        }

        btnConfirm.setOnClickListener {
            val title = etTitle.text.toString()
            val description = etDescription.text.toString()
            if (title.isBlank()) {
                etTitle.error = "제목을 입력해주세요"
                return@setOnClickListener
            }
            val dateStr = selectedDateTime.toString() // "2025-05-23T18:00"
            onConfirm(title, description, dateStr)
            dismiss()
        }

        btnCancel.setOnClickListener { dismiss() }

        // 다이얼로그 width를 화면의 90%로 설정
        dialog.setOnShowListener {
            val width = (resources.displayMetrics.widthPixels * 0.9).toInt()
            dialog.window?.setLayout(width, ViewGroup.LayoutParams.WRAP_CONTENT)
        }

        return dialog
    }
}