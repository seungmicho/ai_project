package com.example.capstoneassistant_na.viewmodel

import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.capstoneassistant_na.data.model.Schedule
import com.example.capstoneassistant_na.data.repository.ScheduleRepository
import kotlinx.coroutines.launch

class ScheduleViewModel : ViewModel() {

    private val repository = ScheduleRepository()

    private val _schedules = MutableLiveData<List<Schedule>>()
    val schedules: LiveData<List<Schedule>> = _schedules

    private val _error = MutableLiveData<String>()
    val error: LiveData<String> = _error

    private val _isLoading = MutableLiveData<Boolean>()
    val isLoading: LiveData<Boolean> = _isLoading

    // 일정 목록 불러오기
    fun loadSchedules() {
        viewModelScope.launch {
            _isLoading.value = true
            try {
                _schedules.value = repository.getSchedules()
            } catch (e: Exception) {
                _error.value = "일정을 불러오지 못했어요: ${e.message}"
            } finally {
                _isLoading.value = false
            }
        }
    }

    // 일정 추가
    fun createSchedule(title: String, description: String, date: String) {
        viewModelScope.launch {
            try {
                repository.createSchedule(title, description, date)
                loadSchedules() // 추가 후 목록 새로고침
            } catch (e: Exception) {
                _error.value = "일정 추가 실패: ${e.message}"
            }
        }
    }

    // 일정 삭제
    fun deleteSchedule(id: Int) {
        viewModelScope.launch {
            try {
                repository.deleteSchedule(id)
                loadSchedules() // 삭제 후 목록 새로고침
            } catch (e: Exception) {
                _error.value = "일정 삭제 실패: ${e.message}"
            }
        }
    }
}