# おノマとペーパー
## 動画URL
[おノマとペーパー](https://youtu.be/YpN9HAPWaxQ)
## 製品概要

### オノマトペ Tech
### 背景（製品開発のきっかけ、課題等）
- 風景画等を何か文字を使用して強調できないかと考えた
- 入力した画像に対応するオノマトペを出力することができるかが課題である。

### 製品説明（具体的な製品の説明）
- 画像を入力し、それに対応した擬音語・擬態語等のオノマトペを画像に付加することで画像を強調することができるWebアプリケーション

### 特長
#### 1. 特長1
- オノマトペを用いた画像の強調という、画像ではなく文字による強調を行うことができる
#### 2. 特長2
- 紙芝居や風景画に自動で強調の文字をつけることができる。

### 解決出来ること
- 画像の文字による加工が、直感的かつ簡潔な表現で可能となる．

### 今後の展望
- 今後は，表示されるオノマトペの数を増やすことと分類器の精度を上げる事で、どんな画像にでも適切に対応する事を目標としたい．

## 開発内容・開発技術
### 活用した技術
- 機械学習

#### API・データ
* 特になし

#### フレームワーク・ライブラリ・モジュール
* bottle(python3)
* Bootstrap
* tensorflow
* keras
* sklearn

#### デバイス
* webサーバーとして用いることができるパソコン

### 研究内容・事前開発プロダクト（任意）
* 私達のチームメンバーである高山 拓夢の"オノマトペの自動意味的推定"の研究の一部成果を用いました。

### 独自開発技術（Hack Dayで開発したもの）
#### 2日間に開発した独自の機能・技術
* オノマトペの自動意味的用法分類器の為の、画像特徴量抽出アルゴリズム
* 離散コサイン変換の一次元変換技術
