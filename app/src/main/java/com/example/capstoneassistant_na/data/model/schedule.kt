package com.example.capstoneassistant_na.data.model

data class Schedule(
    val id: Int,
    val title: String,
    val description: String,
    val date: String,       // "2025-05-23T18:00:00"
    val created_at: String
)

data class ScheduleCreate(
    val title: String,
    val description: String,
    val date: String
)