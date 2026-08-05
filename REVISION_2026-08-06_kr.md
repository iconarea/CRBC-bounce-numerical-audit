# 2026-08-06 수정본 재검증 기록

## 수정 내용

1. 유효장론 절단척도를 임의의 `20 * E_char`가 아니라
   \(\Lambda=\rho_H^{1/4}\)인 문자열/Hagedorn 척도로 설정하는 선택지를 추가했다.
2. 빠른 수축(ekpyrotic)에서 \(w\simeq0\) Hagedorn 말기로 전이하는
   현상론적 \(w(t)\) 프로파일을 검사했다.
3. 수정된 결과와 코드의 한계를 영문 audit 원고에 반영했다.

## 독립 재실행

CPU에서 수정한 `crbc_beyond_horndeski_realization.py`와
`crbc_eft_coefficient_gate.py`를 다시 실행했다.

| 설정 | \(\max(E_{\rm char}/\Lambda)\) | EFT 게이트 |
| --- | ---: | --- |
| \(\rho_H/\rho_c=1\) | 0.5188 | 실패; 9,999점 중 1,840점 위반 |
| \(\rho_H/\rho_c=10^4\) | 0.05188 | 통과 |

두 실행 모두 선택된 계수 프로파일에서 \(Q_s>0\), \(Q_T>0\),
\(c_s^2>0\), \(c_s^2\le1\), \(c_T^2=1\)을 만족했다. 이는 **제공된
계수 프로파일의 이차 안정성 검사**이며, 공변 작용 유도는 아니다.

## 판정

- 직접적인 Hagedorn/문자열 척도 바운스(\(\rho_H=\rho_c\))는 이 코드의
  선언된 EFT 제어 기준을 통과하지 못한다.
- 통과는 바운스 밀도를 Hagedorn 밀도보다 \(10^4\)배 낮게 둘 때만 얻어진다.
  따라서 수정본은 Planck/문자열 척도 바운스의 증거가 아니라,
  **하위 문자열 척도 현상론적 바운스**가 이 특정 게이트에서만 제어된다는
  부정적 제약을 제공한다.
- \(w(t)\) 전이, \(\rho_H/\rho_c\), 절단척도 관계는 여전히 손으로 입력한
  가정이다. 이를 산출하는 공변 beyond-Horndeski/DHOST 작용과 실제 CMB 우도
  분석은 미완료다.

## 재현 명령

```bash
python code/crbc_beyond_horndeski_realization.py --device cpu \
  --p-initial 8 --p-final 1.5 --cutoff-mode string-scale \
  --string-scale-over-rho-c 1 --npz-output /tmp/fail.npz
python code/crbc_eft_coefficient_gate.py /tmp/fail.npz --device cpu
```

두 번째 실행에서 `--string-scale-over-rho-c 10000`을 사용하면 같은 격자
설정에서 게이트를 통과한다.
