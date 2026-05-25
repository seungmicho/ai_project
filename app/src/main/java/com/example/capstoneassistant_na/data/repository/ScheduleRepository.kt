package com.example.capstoneassistant_na.data.repository

import com.example.capstoneassistant_na.data.api.RetrofitClient
import com.example.capstoneassistant_na.data.model.ScheduleCreate

class ScheduleRepository {

    private val api = RetrofitClient.instance

    suspend fun getSchedules() = api.getSchedules()

    suspend fun createSchedule(title: String, description: String, date: String) =
        api.createSchedule(ScheduleCreate(title, description, date))

    suspend fun deleteSchedule(id: Int) = api.deleteSchedule(id)
}