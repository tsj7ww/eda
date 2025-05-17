flowchart TB
    stage1["stage1\nDescription: abc\n\nInputs:\n- in1_1\n- in1_2\n- in1_3\n\nOutputs:\n- out1_1\n- out1_2\n- out1_3"]
    stage2["stage2\nDescription: abc\n\nInputs:\n- in2_1\n- in2_2\n- out1_2\n\nOutputs:\n- out2_1\n- out2_2\n- out2_3"]
    stage3["stage3\nDescription: abc\n\nInputs:\n- out1_2\n- out2_3\n- out1_3\n- out2_2\n\nOutputs:\n- out3_1\n- out3_2\n- out3_3"]
    stage2 -->|"out2_3"| stage3
    stage2 -->|"out2_2"| stage3
    stage1 -->|"out1_2"| stage2
    stage1 -->|"out1_3"| stage3
    stage1 -->|"out1_2"| stage3
    classDef default fill:#f9f9f9,stroke:#333,stroke-width:1px