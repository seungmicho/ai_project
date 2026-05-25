package com.example.capstoneassistant_na.ui.schedule

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import android.widget.Toast
import androidx.fragment.app.Fragment
import androidx.fragment.app.viewModels
import com.example.capstoneassistant_na.R
import com.example.capstoneassistant_na.data.model.Schedule
import com.example.capstoneassistant_na.viewmodel.ScheduleViewModel
import com.kizitonwose.calendar.core.CalendarDay
import com.kizitonwose.calendar.core.firstDayOfWeekFromLocale
import com.kizitonwose.calendar.view.CalendarView
import com.kizitonwose.calendar.view.MonthDayBinder
import com.kizitonwose.calendar.view.ViewContainer
import java.time.LocalDate
import java.time.YearMonth
import java.time.format.DateTimeFormatter
import java.util.Locale

class ScheduleFragment : Fragment() {
    private val viewModel: ScheduleViewModel by viewModels()
    private lateinit var calendarView: CalendarView
    private lateinit var tvCurrentMonth: TextView

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?): View? {
        return inflater.inflate(R.layout.fragment_schedule, container, false)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        calendarView = view.findViewById(R.id.calendarView)
        tvCurrentMonth = view.findViewById(R.id.tvCurrentMonth)

        setupCalendar()
        observeViewModel()
        viewModel.loadSchedules()

        view.findViewById<View>(R.id.btnAddSchedule).setOnClickListener {
            ScheduleAddDialog { title, description, date ->
                viewModel.createSchedule(title, description, date)
            }.show(parentFragmentManager, "AddSchedule")
        }
    }

    private fun setupCalendar() {
        val currentMonth = YearMonth.now()
        calendarView.setup(
            currentMonth.minusMonths(12),
            currentMonth.plusMonths(12),
            firstDayOfWeekFromLocale()
        )
        calendarView.scrollToMonth(currentMonth)
        updateMonthTitle(currentMonth)

        calendarView.monthScrollListener = { month ->
            updateMonthTitle(month.yearMonth)
        }

        calendarView.dayBinder = object : MonthDayBinder<ViewContainer> {
            override fun create(view: View) = ViewContainer(view)
            override fun bind(container: ViewContainer, data: CalendarDay) {
                val textView = container.view.findViewById<TextView>(R.id.calendarDayText)
                textView.text = data.date.dayOfMonth.toString()

                val hasSchedule = viewModel.schedules.value?.any { schedule ->
                    schedule.date.startsWith(data.date.toString())
                } ?: false

                textView.setTextColor(
                    if (hasSchedule)
                        resources.getColor(android.R.color.holo_blue_dark, null)
                    else
                        resources.getColor(android.R.color.black, null)
                )

                container.view.setOnClickListener {
                    val daySchedules = viewModel.schedules.value?.filter { schedule ->
                        schedule.date.startsWith(data.date.toString())
                    }
                    if (!daySchedules.isNullOrEmpty()) {
                        showDaySchedulesDialog(data.date, daySchedules)
                    }
                }
            }
        }
    }

    private fun showDaySchedulesDialog(date: LocalDate, schedules: List<Schedule>) {
        val titles = schedules.map { it.title }.toTypedArray()
        androidx.appcompat.app.AlertDialog.Builder(requireContext())
            .setTitle("${date.monthValue}월 ${date.dayOfMonth}일 일정")
            .setItems(titles) { _, index ->
                showScheduleDetailDialog(schedules[index])
            }
            .setNegativeButton("닫기", null)
            .show()
    }

    private fun showScheduleDetailDialog(schedule: Schedule) {
        androidx.appcompat.app.AlertDialog.Builder(requireContext())
            .setTitle(schedule.title)
            .setMessage(if (schedule.description.isBlank()) "설명 없음" else schedule.description)
            .setPositiveButton("삭제") { _, _ ->
                viewModel.deleteSchedule(schedule.id)
            }
            .setNegativeButton("닫기", null)
            .show()
    }

    private fun observeViewModel() {
        viewModel.schedules.observe(viewLifecycleOwner) {
            calendarView.notifyCalendarChanged()
        }
        viewModel.error.observe(viewLifecycleOwner) { errorMsg ->
            Toast.makeText(requireContext(), errorMsg, Toast.LENGTH_SHORT).show()
        }
    }

    private fun updateMonthTitle(yearMonth: YearMonth) {
        val formatter = DateTimeFormatter.ofPattern("yyyy년 M월", Locale.KOREAN)
        tvCurrentMonth.text = yearMonth.format(formatter)
    }
}