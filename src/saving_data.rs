use crate::{node::Node, State};

impl State {
    pub async fn save_simulation_state(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        compute_buffer: &wgpu::Buffer,
        nodes_x: u32,
        nodes_y: u32,
    ) {
        match Self::save_lbm_data(device, queue, compute_buffer, nodes_x, nodes_y).await {
            Ok(_) => println!("Successfully saved simulation data"),
            Err(e) => eprintln!("Error saving simulation data: {}", e),
        }
    }

    pub async fn save_lbm_data(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        compute_buffer: &wgpu::Buffer,
        nodes_x: u32,
        nodes_y: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Create staging buffer for reading
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: (std::mem::size_of::<Node>() * (nodes_x * nodes_y) as usize) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create command encoder and copy data
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Copy Encoder"),
        });

        encoder.copy_buffer_to_buffer(compute_buffer, 0, &staging_buffer, 0, staging_buffer.size());

        // Submit command
        queue.submit(Some(encoder.finish()));

        // Map the buffer and read data
        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        device.poll(wgpu::Maintain::Wait);

        rx.receive().await.unwrap()?;

        // Get the mapped data
        let data = buffer_slice.get_mapped_range();
        let nodes: &[Node] = bytemuck::cast_slice(&data);

        // Create CSV writers
        let mut full_writer = csv::Writer::from_path("lbm_data.csv")?;
        let mut centerline_writer = csv::Writer::from_path("centerline_data.csv")?;

        // Write headers
        full_writer.write_record(&[
            "x", "y", "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "density", "ux", "uy",
        ])?;

        centerline_writer.write_record(&["position", "u_velocity", "v_velocity"])?;

        // LBM velocity vectors
        let c_x: [f32; 9] = [0.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, -1.0, 1.0];
        let c_y: [f32; 9] = [0.0, 0.0, 1.0, 0.0, -1.0, 1.0, 1.0, -1.0, -1.0];

        // Write full data
        for y in 0..nodes_y {
            for x in 0..nodes_x {
                let idx = (y * nodes_x + x) as usize;
                let node = &nodes[idx];

                // Calculate macroscopic quantities
                let density: f32 = node.distribution.iter().sum();
                let mut momentum_x = 0.0;
                let mut momentum_y = 0.0;

                for i in 0..9 {
                    momentum_x += c_x[i] * node.distribution[i];
                    momentum_y += c_y[i] * node.distribution[i];
                }

                let ux = momentum_x / density;
                let uy = momentum_y / density;

                full_writer.write_record(&[
                    x.to_string(),
                    y.to_string(),
                    node.distribution[0].to_string(),
                    node.distribution[1].to_string(),
                    node.distribution[2].to_string(),
                    node.distribution[3].to_string(),
                    node.distribution[4].to_string(),
                    node.distribution[5].to_string(),
                    node.distribution[6].to_string(),
                    node.distribution[7].to_string(),
                    node.distribution[8].to_string(),
                    density.to_string(),
                    ux.to_string(),
                    uy.to_string(),
                ])?;
            }
        }

        // Write centerline data
        // Vertical centerline (u velocity)
        let mid_x = nodes_x / 2;
        for y in 0..nodes_y {
            let idx = (y * nodes_x + mid_x) as usize;
            let node = &nodes[idx];

            let density: f32 = node.distribution.iter().sum();
            let mut momentum_x = 0.0;
            for i in 0..9 {
                momentum_x += c_x[i] * node.distribution[i];
            }
            let u_velocity = momentum_x / density;

            centerline_writer.write_record(&[
                (y as f32 / (nodes_y - 1) as f32).to_string(),
                u_velocity.to_string(),
                "NaN".to_string(),
            ])?;
        }

        // Horizontal centerline (v velocity)
        let mid_y = nodes_y / 2;
        for x in 0..nodes_x {
            let idx = (mid_y * nodes_x + x) as usize;
            let node = &nodes[idx];

            let density: f32 = node.distribution.iter().sum();
            let mut momentum_y = 0.0;
            for i in 0..9 {
                momentum_y += c_y[i] * node.distribution[i];
            }
            let v_velocity = momentum_y / density;

            centerline_writer.write_record(&[
                (x as f32 / (nodes_x - 1) as f32).to_string(),
                "NaN".to_string(),
                v_velocity.to_string(),
            ])?;
        }

        // Unmap buffer
        drop(data);
        staging_buffer.unmap();

        Ok(())
    }
}
