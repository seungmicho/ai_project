package com.example.capstoneassistant_na.data.api

import com.example.capstoneassistant_na.data.model.Schedule
import com.example.capstoneassistant_na.data.model.ScheduleCreate
import retrofit2.http.*

interface ApiService {

    // 일정 전체 조회
    @GET("schedules/")
    suspend fun getSchedules(): List<Schedule>

    // 일정 추가
    @POST("schedules/")
    suspend fun createSchedule(@Body body: ScheduleCreate): Schedule

    // 일정 삭제
    @DELETE("schedules/{id}")
    suspend fun deleteSchedule(@Path("id") id: Int): Map<String, String>
}