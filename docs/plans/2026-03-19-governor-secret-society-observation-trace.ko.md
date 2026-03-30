# 총독 비밀결사 관찰/트레이스 구현 계획

이 문서는 `governor_primitive`의 observation scroll, secret-society branch, Rich trace behavior를 개선하는 계획입니다.

## 목표

- 총독 primitive가 최소 한 번은 downward observation scroll을 강제
- secret-society appointment branch를 추가하고 promotion으로 합류 가능하게 만들기
- governor 관련 Rich trace 동작을 tests로 고정

## 아키텍처

- 모든 로직은 `GovernorProcess` 안에 유지
- observation, branch selection, branch merge를 코드가 소유
- 별도의 trace 시스템을 만들지 않고 `turn_executor.py`의 runtime trace feed를 재사용

## 구현 포인트

- governor observation regression tests 추가
- 최소 1회 downward scan rule 도입
- no-new-candidates / end-of-list 확인 후에만 선택 단계로 전환

이 페이지는 계획 요약이며, 원문은 테스트 주도 구현 단계까지 자세히 포함합니다.
