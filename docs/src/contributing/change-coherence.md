---
change_coherence:
  scenarios:
    small_clear: direct
    related_finding: sibling_search_before_fix
    unstable_contract: contract_replan_required
---

# Risk-based Change Coherence / リスクベースの変更整合性

Use this procedure for ambiguous, high-risk, cross-cutting, or review-driven
changes. Its purpose is to keep the contract, ownership, invariant families,
scope, implementation, and review evidence coherent while the change evolves.
曖昧、高リスク、cross-cutting、またはreviewを起点とする変更では、この手順を使用します。
変更の進行中もcontract、ownership、invariant family、scope、実装、review evidenceを
一貫させることが目的です。

Repository agents reach this guide through the
[`wandas-change-coherence`](https://github.com/kasahart/wandas/blob/main/.agents/skills/wandas-change-coherence/SKILL.md)
Skill. This guide is the canonical procedure; Skills and vendor adapters only
trigger or route to it.
Repository agentは`wandas-change-coherence` Skillからこのguideへ到達します。このguideが
手順の正本であり、Skillとvendor adapterはtriggerまたはrouteだけを担当します。

## Trigger by risk, not by task count / task数ではなくriskで発火する

Classify the change before imposing process. Reassess whenever findings or the
implementation reveal a wider boundary.
processを課す前に変更を分類します。findingや実装から広いboundaryが判明した場合は再評価します。

| Change state / 変更状態 | Route / 経路 |
| --- | --- |
| One local owner, explicit acceptance, no public or architectural ambiguity / ownerが局所的でacceptanceが明確、公開・architecture上の曖昧さがない | Implement directly; no additional artifact or independent review is required / 直接実装する。追加artifactやindependent reviewは不要 |
| A concrete finding may share state, cause, or boundary with other behavior / 具体的findingが他behaviorとstate・原因・boundaryを共有しうる | Classify the invariant family and perform a bounded sibling search before patching / invariant familyを分類し、patch前にbounded sibling searchを行う |
| Public behavior, architecture, ownership, compatibility, acceptance, or scope is ambiguous or materially moving / 公開behavior、architecture、ownership、compatibility、acceptance、scopeが曖昧またはmaterialに変動 | Capture the changing decisions where reviewers can see them and apply the replan boundary / 変動する判断をreviewerが確認できる場所へ記録し、replan boundaryを適用 |

Risk is about consequence and uncertainty, not diff size. A one-line public
contract change can be high-risk; a large mechanical rename with one owner can
remain straightforward.
riskはdiff sizeではなく影響と不確実性で決まります。1行の公開contract変更が高リスクになり、
ownerが1つの大きな機械的renameが明確な変更に留まることもあります。

## Establish the current contract / 現在のcontractを確立する

Before implementing a triggered change, make the following small set of
decisions explicit. Capture them in task notes or the PR description when the
answers may evolve; no repository-wide form is required.
発火対象の変更を実装する前に、次の少数の判断を明示します。回答が変動しうる場合はtask noteまたは
PR descriptionへ記録しますが、repository-wideなformは必須ではありません。

1. State the observable acceptance boundary: what must be true for the change
   to be complete, and what evidence can disprove it.
   観測可能なacceptance boundary、つまり完了時に成立すべきことと、それを反証できるevidenceを定義します。
2. Separate behavior into `supported`, `restricted`, and `invalid`. A restricted
   state has an explicit constraint or adapter; an invalid state is rejected
   rather than silently normalized.
   behaviorを`supported`、`restricted`、`invalid`へ分けます。restricted stateには明示的な
   constraintまたはadapterがあり、invalid stateはsilentにnormalizeせずrejectします。
3. Name the owner of each changed state or behavior. Consumers may expose or
   transform owned state, but must not become a synchronized second owner.
   変更するstateやbehaviorごとにownerを特定します。consumerはowned stateを公開・変換できますが、
   同期が必要な第2のownerにはしません。
4. Record initial scope and current scope by responsibility or public boundary,
   not merely by file list.
   initial scopeとcurrent scopeをfile listだけでなくresponsibilityまたはpublic boundaryで記録します。

Acceptance, implementation, tests, documentation, and tracking must describe
the same current contract. Passing tests do not stabilize an undecided contract.
acceptance、実装、test、documentation、trackingは同じ現在のcontractを表す必要があります。
test通過は未決定のcontractをstableにはしません。

## Group findings into invariant families / findingをinvariant familyにまとめる

For every substantive finding, first ask whether it is a local defect or an
instance of a missing shared invariant. Give the family a cause-oriented name,
not a symptom or file name.
substantiveなfindingごとに、局所欠陥か、欠けたshared invariantの一例かを最初に判断します。
familyには症状やfile名ではなく原因を表す名前を付けます。

A useful family statement identifies:
有用なfamily statementは次を特定します。

- the state or behavior whose validity is at risk;
  validityにriskがあるstateまたはbehavior。
- its single owner and the path that bypasses or contradicts that owner;
  single ownerと、それを迂回または矛盾させるpath。
- the supported, restricted, or invalid class affected;
  影響するsupported、restricted、invalidの区分。
- the invariant that would make every family member correct.
  family全体を正しくするinvariant。

Do not treat sibling examples as unrelated tickets until this classification is
done. Conversely, do not expand a family merely because symptoms look similar;
shared cause or ownership is the boundary.
この分類が終わるまでsibling exampleを無関係なticketとして扱いません。一方、症状が似ているだけで
familyを拡張せず、共有causeまたはownershipをboundaryにします。

## Perform a bounded sibling search / bounded sibling searchを行う

Search before fixing, but stop at an evidence-based boundary. Start from the
owner and inspect the routes that create, change, persist, transform, or expose
the same state:
修正前に探索しますが、evidenceに基づくboundaryで停止します。ownerから始め、同じstateを生成、変更、
永続化、変換、公開する経路を確認します。

- construction and deserialization;
  constructionとdeserialization。
- mutation, replacement, and copy paths;
  mutation、replacement、copy path。
- serialization and persistence;
  serializationとpersistence。
- transformation and domain transitions;
  transformationとdomain transition。
- public methods, exports, adapters, and validation boundaries.
  public method、export、adapter、validation boundary。

Record both the searched boundaries and what was actually checked. Stop when a
route has a different owner or contract, cannot carry the affected state, or is
outside the accepted scope with an explicit tracking decision. Add a family-level
test when one assertion matrix, parameterization, or shared helper can protect
the failure family; keep focused examples when they prove distinct contracts.
探索したboundaryと実際に確認した対象の両方を記録します。異なるowner・contractを持つ経路、対象stateを
運べない経路、または明示的tracking判断によりaccepted scope外となる経路で停止します。1つのassertion
matrix、parameterization、shared helperでfailure familyを保護できる場合はfamily-level testを追加し、
異なるcontractを証明する例はfocused exampleとして残します。

## Use the explicit replan boundary / 明示的replan boundaryを使う

The following are replan signals, not ordinary requests for another local patch:
次は通常の局所patch依頼ではなくreplan signalです。

- related findings recur around the same invariant family;
  同じinvariant familyの周辺でrelated findingが繰り返される。
- each local fix exposes another required local fix or increases branching;
  局所修正ごとに別の局所修正が必要となる、または分岐が増える。
- an implemented contract or guard is withdrawn later;
  実装済みcontractまたはguardを後から撤回する。
- public behavior, architecture, ownership, compatibility, acceptance, or scope
  changes materially;
  公開behavior、architecture、ownership、compatibility、acceptance、scopeがmaterialに変わる。
- production changes spread into multiple unplanned responsibility domains;
  production変更が当初未計画の複数responsibility domainへ波及する。
- validation passes while the acceptance contract remains uncertain or moving.
  validationは通過するがacceptance contractが未確定または変動している。

Signals require an explicit decision and rationale; no fixed count automatically
stops work. Continue locally only when the current contract and owners remain
stable, the sibling family is bounded, and one coherent fix covers it. Otherwise
stop implementation, revise the contract and acceptance boundary, update scope
and ownership, invalidate stale evidence, and then plan a new coherent patch.
signalには明示的なdecisionとrationaleが必要であり、固定回数だけで自動停止しません。現在のcontractと
ownerがstableで、sibling familyがboundedであり、1つのcoherent fixが全体をcoverする場合だけ局所修正を
継続します。それ以外では実装を止め、contractとacceptance boundary、scope、ownershipを更新し、古い
evidenceを無効化してから新しいcoherent patchを計画します。

## Review invariants, not only lines / lineだけでなくinvariantをreviewする

Independent review is optional and should be selected for useful risk reduction,
not as a mandatory pipeline stage. When used for a triggered change, the reviewer:
independent reviewは任意であり、必須pipeline stageではなくrisk低減に有効な場合に選びます。
発火対象の変更をreviewする場合、reviewerは次を行います。

1. Classify each finding as a local defect or invariant-family evidence.
   各findingを局所欠陥またはinvariant-family evidenceへ分類する。
2. Inspect the sibling-search boundary and symmetry across construction,
   mutation, serialization, transformation, and public access where applicable.
   該当するconstruction、mutation、serialization、transformation、public accessの対称性と
   sibling-search boundaryを確認する。
3. Check whether tests protect the failure family rather than only the reported
   example, and whether new branches or compatibility state keep accumulating.
   testが報告例だけでなくfailure familyを保護するか、新しい分岐やcompatibility stateが増え続けて
   いないか確認する。
4. Return `FIX_REQUIRED` for a stable contract with bounded implementation work,
   or `CONTRACT_REPLAN_REQUIRED` when a material boundary is unstable.
   stable contractにboundedな実装作業が残る場合は`FIX_REQUIRED`、material boundaryが不安定な場合は
   `CONTRACT_REPLAN_REQUIRED`を返す。

## Gate external review on current evidence / external reviewを現在のevidenceでgateする

For a triggered change, request external review only when all of the following
describe the current head and contract revision:
発火対象の変更では、次のすべてがcurrent headとcontract revisionを表す場合だけexternal reviewを
依頼します。

- current contract and responsibility-level scope are explained;
  現在のcontractとresponsibility単位のscopeが説明されている。
- implementation, tests, documentation, and tracking are aligned or explicitly
  not applicable;
  実装、test、documentation、trackingがalignedまたは明示的にnot applicableである。
- known findings are resolved or tracked by invariant family after bounded sibling searches;
  known findingがbounded sibling search後にinvariant family単位で解決またはtrackingされている。
- no material open decision is hidden from reviewers;
  materialなopen decisionがreviewerから隠されていない。
- deferred work has durable tracking, validation evidence lists failures and
  justified skips, and residual risk is stated;
  deferred workにdurable trackingがあり、validation evidenceがfailureと理由付きskipを示し、
  residual riskが明示されている。
- evidence head and contract revision match the current head and revision.
  evidence headとcontract revisionが現在値に一致する。

If architecture or scope changed materially after earlier review, revise the
contract revision and collect new evidence. Do not reuse the earlier readiness
claim even when CI still passes.
以前のreview後にarchitectureまたはscopeがmaterialに変わった場合、contract revisionを更新し、
新しいevidenceを収集します。CIが通り続けていても以前のreadiness claimを再利用しません。

## Keep mechanical checks deterministic / 機械的検査を決定論的に保つ

Do not encode this judgment-heavy procedure in an executable policy engine.
Harness architecture tests instead verify that the Skill routes here, this page
is the single procedure owner, vendor adapters stay thin, local links and
frontmatter resolve, and the three representative scenario classes remain
structurally distinct. CI, linters, type checks, documentation builds, and
domain tests own facts they can actually observe.
判断の多いこのprocedureを実行可能なpolicy engineへ符号化しません。代わりにharness architecture
testが、Skillからこのpageへのroute、このpageによるprocedureのsingle ownership、thinなvendor
adapter、local linkとfrontmatter、3つの代表scenario classの構造的な区別を検査します。CI、lint、
type check、documentation build、domain testは、それぞれが実際に観測できる事実をownerとします。

Contract stability, sibling completeness, and materiality remain review
judgments backed by explicit evidence. A test must not turn a self-reported
placeholder or duplicated state into a readiness claim.
contract stability、siblingの網羅性、materialityは、明示的evidenceに基づくreview判断として残します。
testは自己申告のplaceholderやduplicated stateをreadiness claimへ変換してはなりません。
