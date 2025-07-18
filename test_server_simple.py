#!/usr/bin/env python3
import requests
import json
import sys

print("🚀 서버 빠른 테스트")
print("=" * 30)

BASE_URL = "http://localhost:8002"

# 1. 서버 상태 확인
print("1. 서버 상태 확인")
try:
    response = requests.get(f"{BASE_URL}/")
    print(f"   상태코드: {response.status_code}")
    if response.status_code == 200:
        try:
            data = response.json()
            print(f"   서버 상태: {data.get('status', 'unknown')}")
            print(f"   버전: {data.get('version', 'unknown')}")
        except:
            print(f"   응답: {response.text[:100]}...")
    else:
        print(f"   응답: {response.text}")
except Exception as e:
    print(f"   오류: {e}")

# 2. 헬스 체크
print("\n2. 헬스 체크")
try:
    response = requests.get(f"{BASE_URL}/api/health")
    print(f"   상태코드: {response.status_code}")
    if response.status_code == 200:
        try:
            data = response.json()
            print(f"   헬스 상태: {data.get('status', 'unknown')}")
        except:
            print(f"   응답: {response.text[:100]}...")
    else:
        print(f"   응답: {response.text}")
except Exception as e:
    print(f"   오류: {e}")

print("\n" + "=" * 30)
print("✅ 테스트 완료!")