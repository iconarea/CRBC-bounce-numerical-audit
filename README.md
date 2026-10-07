# crbc_bounce_numerical_audit — 암호화 보관 저장소

> **이 안내문은 `encpush` 가 푸시할 때마다 자동으로 다시 만든다.** 여기에 손으로 쓴 내용은 다음 푸시에서 사라진다 —
> 문안을 바꾸려면 `~/bin/encpush` 의 템플릿을 고친다.
> 아래 문장은 **이 저장소(암호화 보관 등급)에만** 적용된다. 평문으로 두는 저장소가 따로 있고, 어느 저장소가
> 어느 등급인지는 `ai-cowork-os/00-COMMON/GITHUB-ENCRYPTION-CONVENTION.md`(저장 규칙)가 정본이다.
> 정보 자체의 등급(L0~L3)은 `ai-business-os/00-COMMON/SENSITIVITY.md` 가 정본이다.

이 저장소의 **평문 내용은 GitHub 에 두지 않는다.** 전체 이력(모든 브랜치·태그)은
release `latest` 의 자산 `CRBC-bounce-numerical-audit.bundle.zst.gpg` 하나에 들어 있고, 푸시할 때마다 덮어쓴다.

- 형식: `git bundle --all` → `zstd -19` → `gpg --symmetric AES256`
- 복호화 키: 로컬 `~/.config/encpush/key` (GitHub 에 없음)
- 마지막 갱신: 2026-10-08T04:50:01+09:00 · sha256 앞자리 `f62012941d837ca7` · 크기 5.4M

## 복원

```bash
gh release download latest -R iconarea/CRBC-bounce-numerical-audit -p '*.bundle.zst.gpg'
gpg --batch --decrypt --passphrase-file ~/.config/encpush/key CRBC-bounce-numerical-audit.bundle.zst.gpg | zstd -d > CRBC-bounce-numerical-audit.bundle
git clone CRBC-bounce-numerical-audit.bundle crbc_bounce_numerical_audit
```

평문 원본은 로컬과 자체 서버 원격(debian·rocky)에 있다.
