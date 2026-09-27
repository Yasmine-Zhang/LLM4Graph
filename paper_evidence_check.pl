#!/usr/bin/env perl
use strict;
use warnings;
use Digest::SHA qw(sha256_hex);
use JSON::PP;

my $root = 'output/analysis';
my $pilot = "$root/amazon_equal_fact_learner_pilot_20260925_v0";

sub bytes {
    my ($path) = @_;
    open my $handle, '<:raw', $path or die "Missing evidence $path: $!\n";
    local $/;
    return <$handle>;
}

sub json {
    return decode_json(bytes(shift));
}

sub expect {
    my ($actual, $expected, $name) = @_;
    die "$name: expected $expected, got $actual\n" unless defined($actual) && $actual eq $expected;
}

sub exact_one_sided {
    my ($fixes, $harms) = @_;
    my $discordant = $fixes + $harms;
    return 1 if $discordant == 0;
    my ($combination, $tail) = (1, 0);
    for my $wins (0 .. $discordant) {
        $tail += $combination if $wins >= $fixes;
        $combination *= ($discordant - $wins) / ($wins + 1) if $wins < $discordant;
    }
    return $tail / (2 ** $discordant);
}

sub close_to {
    my ($actual, $expected, $name) = @_;
    die "$name: expected $expected, got $actual\n"
        unless defined($actual) && abs($actual - $expected) <= 1e-9;
}

sub verify_frozen {
    my ($folder, $count, $primary, $expected_hits, $expected_pairs, $expected_significant) = @_;
    my $file = "$root/$folder/results.json";
    my $result = json($file);
    my $actions = "$root/$folder/" . ($folder =~ /^disjoint/ ? 'adapted_actions_frozen.json' : 'fusion_actions_frozen.json');
    expect(sha256_hex(bytes($actions)), $result->{actions_hash}, "$folder action hash");
    expect(scalar @{$result->{records}}, $count, "$folder record count");
    my %records = map { $_->{id} => $_ } @{$result->{records}};
    expect(scalar keys %records, $count, "$folder unique query IDs");
    my %summaries = map { $_->{method} => $_ } @{$result->{summaries}};
    for my $method (sort keys %$expected_hits) {
        my $summary = $summaries{$method} or die "No $folder summary for $method\n";
        expect($summary->{count}, $count, "$folder $method denominator");
        expect($summary->{hits}, $expected_hits->{$method}, "$folder $method summary");
        my $sum = 0;
        for my $record (@{$result->{records}}) {
            my $hit = $record->{hits}{$method};
            die "Invalid $folder hit for $method\n" unless defined $hit && ($hit == 0 || $hit == 1);
            $sum += $hit;
        }
        expect($sum, $summary->{hits}, "$folder $method row sum");
    }
    my @comparators = @{$result->{comparators}};
    my $inference = $result->{inference};
    expect(scalar @comparators, $folder =~ /^disjoint/ ? 15 : 10, "$folder Holm family size");
    expect(scalar @{$inference->{holm_p}}, scalar @comparators, "$folder p-value count");
    my (@raw_p, @adjusted_p);
    for my $index (0 .. $#comparators) {
        my $name = $comparators[$index];
        my ($fixes, $harms) = (0, 0);
        for my $record (@{$result->{records}}) {
            my $difference = $record->{hits}{$primary} - $record->{hits}{$name};
            $fixes++ if $difference == 1;
            $harms++ if $difference == -1;
        }
        expect($inference->{fixes}[$index], $fixes, "$folder archived fixes vs $name");
        expect($inference->{harms}[$index], $harms, "$folder archived harms vs $name");
        expect($inference->{net}[$index], $fixes - $harms, "$folder archived net vs $name");
        if (exists $expected_pairs->{$name}) {
            my ($expected_fixes, $expected_harms) = @{$expected_pairs->{$name}};
            expect($fixes, $expected_fixes, "$folder fixes vs $name");
            expect($harms, $expected_harms, "$folder harms vs $name");
        }
        $raw_p[$index] = exact_one_sided($fixes, $harms);
        close_to($inference->{p}[$index], $raw_p[$index], "$folder exact p vs $name");
    }
    my $running = 0;
    my @order = sort { $raw_p[$a] <=> $raw_p[$b] || $a <=> $b } 0 .. $#comparators;
    for my $rank (0 .. $#order) {
        my $index = $order[$rank];
        my $candidate = (scalar(@order) - $rank) * $raw_p[$index];
        $running = $candidate if $candidate > $running;
        $adjusted_p[$index] = $running > 1 ? 1 : $running;
    }
    my $significant = 0;
    for my $index (0 .. $#comparators) {
        close_to($inference->{holm_p}[$index], $adjusted_p[$index],
            "$folder Holm p vs $comparators[$index]");
        $significant++ if $adjusted_p[$index] <= .05 && $inference->{net}[$index] > 0;
    }
    expect($significant, $expected_significant, "$folder Holm positives");
    print "PASS $folder: $count records, $significant adjusted positive comparisons\n";
    return \%records;
}

my $h = verify_frozen('multiview_transfer384_20260922', 384, 'graph_fusion',
    { graph_fusion => 278, text_fusion => 264, shuffled_fusion => 263, top1 => 279 },
    { text_fusion => [15, 1], shuffled_fusion => [17, 2], top1 => [13, 14] }, 4);
my $i = verify_frozen('disjoint_transfer768_20260923', 768, 'adapted_graph',
    { adapted_graph => 568, adapted_text => 549, adapted_shuffled => 552,
      top1 => 573, graph_fusion => 563 },
    { adapted_text => [24, 5], adapted_shuffled => [21, 5], top1 => [20, 25] }, 5);

my $protocol = json("$pilot/protocol.json");
my $actions = json("$pilot/actions.json");
my $h_actions = json("$root/multiview_transfer384_20260922/fusion_actions_frozen.json");
my %h_actions = map { $_->{id} => $_ } @{$h_actions->{records}};
expect(scalar keys %h_actions, 384, 'unique frozen H actions');
my $scoring = json("$pilot/scoring_preflight.json");
my $development = json("$pilot/scored_development.json");
open my $gate_handle, '-|', 'perl', "$root/label_blind_action_gate.pl",
    "$pilot/protocol.json", "$pilot/actions.json" or die "Cannot run frozen action gate: $!\n";
local $/;
my $gate_output = <$gate_handle>;
close $gate_handle or die "Frozen action gate failed\n";
my $recomputed_gate = decode_json($gate_output);
my $canonical = JSON::PP->new->canonical;
expect($canonical->encode($recomputed_gate), $canonical->encode($scoring->{gate}),
    'pilot preflight gate matches recomputed actions');
expect($canonical->encode($recomputed_gate), $canonical->encode($development->{gate}),
    'pilot scored gate matches recomputed actions');
expect($protocol->{source_hash}, sha256_hex(bytes("$pilot/source.py")), 'pilot source hash');
for my $source (sort keys %{$protocol->{input_hashes}}) {
    expect(sha256_hex(bytes($source)), $protocol->{input_hashes}{$source}, "pilot input $source");
}
for my $source (sort keys %{$scoring->{inputs}}) {
    expect(sha256_hex(bytes($source)), $scoring->{inputs}{$source}, "scoring input $source");
}
expect($scoring->{source_hash}, sha256_hex(bytes("$pilot/scoring_source.py")), 'scorer source hash');
expect($actions->{protocol_hash}, sha256_hex(bytes("$pilot/protocol.json")), 'pilot protocol hash');
expect($development->{scoring_preflight_hash}, sha256_hex(bytes("$pilot/scoring_preflight.json")), 'scoring preflight hash');
die "Pilot action file claims to contain labels\n" if $actions->{labels_read};
die "Pilot scoring gate did not pass\n" unless $development->{gate}{scoring_permitted_by_action_gate};
expect(scalar @{$actions->{records}}, 384, 'pilot query count');
my %fit = map { $_ => 1 } @{$protocol->{fit_ids}};
expect(scalar keys %fit, 768, 'pilot fit query count');
my ($same, $different) = (0, 0);
for my $index (0 .. 383) {
    my $row = $actions->{records}[$index];
    expect($row->{id}, $protocol->{selected_ids}[$index], 'pilot frozen query order');
    die "Pilot fitting/evaluation query overlap\n" if $fit{$row->{id}};
    my $original = $h->{$row->{id}} or die "Pilot row absent from frozen H\n";
    $same++ if $row->{structured_text} eq $row->{graph_fusion};
    $different++ if $row->{structured_text} ne $row->{graph_fusion};
    my $frozen = $h_actions{$row->{id}} or die "Missing frozen H action\n";
    expect($row->{graph_fusion}, $frozen->{choices}{graph_fusion}, 'original graph action');
    expect($row->{top_link}, $frozen->{choices}{top1}, 'original top-link action');
}
expect($same, 358, 'identical H actions');
expect($different, 26, 'different H actions');
for my $method (qw(structured_text graph_fusion top_link)) {
    expect($development->{hits}{$method}, $method eq 'top_link' ? 279 : 278, "pilot $method Hit\@1");
}
for my $method (qw(graph_fusion top_link)) {
    my $entry = $development->{comparisons}{$method};
    my ($fixes, $harms, $disagreements) = $method eq 'top_link' ? (13, 14, 73) : (4, 4, 26);
    expect($entry->{fixes}, $fixes, "pilot $method fixes");
    expect($entry->{harms}, $harms, "pilot $method harms");
    expect($entry->{action_disagreements}, $disagreements, "pilot $method action differences");
    expect($entry->{net}, $fixes - $harms, "pilot $method net");
}
print "PASS H post-hoc pilot: disjoint fit IDs, 278/278/279, 358 same / 26 different actions\n";

my $candidate = "$root/amazon_candidate_cohort_20260925_v0";
my $queue = json("$candidate/protocol.json");
expect(sha256_hex(bytes("$candidate/source.py")), $queue->{source_sha256}, 'candidate source hash');
for my $source (sort keys %{$queue->{input_sha256}}) {
    expect(sha256_hex(bytes($source)), $queue->{input_sha256}{$source}, "candidate input $source");
}
for my $audit (sort keys %{$queue->{prior_audits_sha256}}) {
    for my $file (sort keys %{$queue->{prior_audits_sha256}{$audit}}) {
        my $path = "$root/$audit/$file";
        expect(sha256_hex(bytes($path)), $queue->{prior_audits_sha256}{$audit}{$file}, "candidate audit $path");
    }
}
expect(scalar @{$queue->{selected_ids}}, 384, 'candidate queue length');
my %candidate_ids = map { $_ => 1 } @{$queue->{selected_ids}};
expect(scalar keys %candidate_ids, 384, 'unique candidate queue IDs');
die "Candidate queue overlaps exposed H/I or pilot fit\n"
    if grep { exists($h->{$_}) || exists($i->{$_}) || exists($fit{$_}) } keys %candidate_ids;
expect($queue->{paid_call_cap}, 0, 'candidate paid-call cap');
die "Candidate query or answer access was enabled\n"
    if $queue->{answers_access_permitted} || $queue->{query_text_access_permitted};
expect($queue->{action_gate}{minimum_net_gain_hits}, 5, 'candidate action gate margin');
expect(join(',', @{$queue->{action_gate}{comparators}}), 'text,top_link', 'candidate strong controls');
print "PASS 384 candidate IDs, H/I/fit disjointness, source/input/audit hashes and zero-access gate\n";

my $resource = "$root/amazon_equal_fact_resource_20260925_v0";
my $resource_protocol = json("$resource/protocol.json");
my $resource_result = json("$resource/results.json");
expect(sha256_hex(bytes("$resource/source.py")), $resource_protocol->{source_sha256}, 'resource source hash');
for my $source (sort keys %{$resource_protocol->{input_sha256}}) {
    expect(sha256_hex(bytes($source)), $resource_protocol->{input_sha256}{$source}, "resource input $source");
}
expect(sha256_hex(bytes("$resource/protocol.json")), $resource_result->{input_sha256}, 'resource protocol hash');
expect($resource_result->{queries}, 384, 'resource H query count');
expect($resource_result->{candidate_rows}, 7680, 'resource H candidate rows');
die "Resource evidence asserts labels or action mismatch\n"
    if $resource_result->{labels_read} || !$resource_result->{both_arms_match_frozen_actions};
for my $measure (
    ['text_serialization_parse_seconds', '0.55070'],
    ['graph_cached_predict_seconds', '0.02310'],
    ['text_cached_predict_seconds', '0.02334'],
) {
    my ($name, $reported) = @$measure;
    my @runs = sort { $a <=> $b } @{$resource_result->{timings}{$name}};
    expect(scalar @runs, $resource_protocol->{repetitions}, "$name run count");
    die "Invalid $name timing\n" if grep { $_ <= 0 } @runs;
    expect(sprintf('%.5f', $runs[3]), $reported, "$name median");
}
print "PASS exposed-H resource microbenchmark hashes, seven runs, medians and action accounting\n";

my $construction = "$root/amazon_equal_fact_construction_20260925_v0";
my $construction_protocol = json("$construction/protocol.json");
my $construction_result = json("$construction/results.json");
expect(sha256_hex(bytes("$construction/source.py")), $construction_protocol->{source_sha256}, 'construction source hash');
for my $source (sort keys %{$construction_protocol->{input_sha256}}) {
    expect(sha256_hex(bytes($source)), $construction_protocol->{input_sha256}{$source}, "construction input $source");
}
expect(sha256_hex(bytes("$construction/protocol.json")), $construction_result->{protocol_sha256}, 'construction protocol hash');
expect($construction_result->{queries}, 384, 'construction query count');
expect($construction_result->{candidate_rows}, 7680, 'construction candidate rows');
expect($construction_result->{graph_scalars_verified}, 46080, 'construction graph scalar count');
die "Construction result claims label access\n" if $construction_result->{labels_read};
for my $measure (['graph_seconds', '0.126542'], ['text_seconds', '0.113578']) {
    my ($name, $reported) = @$measure;
    my @runs = sort { $a <=> $b } @{$construction_result->{times}{$name}};
    expect(scalar @runs, $construction_protocol->{repetitions}, "$name run count");
    die "Invalid $name timing\n" if grep { $_ <= 0 } @runs;
    expect(sprintf('%.6f', $runs[3]), $reported, "$name median");
}
print "PASS exposed-H paired construction source/input hashes, 46,080 graph scalars and seven-run medians\n";

my $paper = bytes('paper.md');
my ($h_table) = $paper =~ /### 8\.4 冻结H：[^\n]*\n(.*?)### 8\.5 /s;
my ($i_table) = $paper =~ /### 8\.7 [^\n]*\n(.*?)## 9\./s;
die "Cannot locate frozen H/I tables in manuscript\n" unless defined($h_table) && defined($i_table);
for my $row (
    '| Top-link，双排名平局处理 | 269 | **279** |',
    '| Text fusion | 266 | 264 |',
    '| **Graph fusion，主体方法** | **279** | **278** |',
    '| Shuffled fusion | 265 | 263 |',
) {
    die "Frozen H table changed: $row\n" unless index($h_table, $row) >= 0;
}
for my $row (
    '| Top-link support | **573** |',
    '| 原18维图融合 | 563 |',
    '| 适配文本 / 适配图置乱融合 | 549 / 552 |',
    '| **新增20维适配图融合** | **568** |',
) {
    die "Frozen I table changed: $row\n" unless index($i_table, $row) >= 0;
}
for my $file (qw(66_AMAZON_EQUAL_FACT_RULE_REPLAY.md 67_AMAZON_EQUAL_FACT_FEATURE_REPLAY.md
                 68_AMAZON_EQUAL_FACT_LEARNER_DEVELOPMENT.md)) {
    die "Manuscript missing audit $file\n" unless index($paper, "$root/$file") >= 0;
}
die "Manuscript missing exposed-H qualification\n"
    unless $paper =~ /H标签已经暴露/ && $paper =~ /不能据同分声称等效/;
die "Manuscript lacks scoped follow-up or resource limitations\n"
    unless index($paper, "$root/70_CANDIDATE_COHORT_AND_EQUAL_FACT_RESOURCE.md") >= 0 &&
           $paper =~ /不是同事实等资源端到端对照/ && $paper =~ /没有新队列确认结果/ &&
           $paper =~ /0\.02310\/0\.02334秒/ && $paper =~ /0\.55070秒/ &&
           $paper =~ /0\.126542秒、文本侧0\.113578秒/;
print "PASS manuscript H/I tables, evidence links and non-equivalence qualification\n";