package com.example.capstoneassistant_na.data.api

import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory

object RetrofitClient {
    // TODO: 백엔드 서버 주소로 교체 필요
    // 같은 WiFi 환경: "http://서버PC_IP:8000/"
    // 배포 서버: "http://실서버주소:8000/"
    private const val BASE_URL = "http://localhost:8000/"

    val instance: ApiService by lazy {
        Retrofit.Builder()
            .baseUrl(BASE_URL)
            .addConverterFactory(GsonConverterFactory.create())
            .build()
            .create(ApiService::class.java)
    }
}