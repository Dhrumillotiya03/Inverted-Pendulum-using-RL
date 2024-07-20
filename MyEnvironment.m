classdef MyEnvironment < rl.env.MATLABEnvironment
    %MYENVIRONMENT: Template for defining custom environment in MATLAB.    
    
    %% Properties (set properties' attributes accordingly)
    properties
        % Specify and initialize environment's necessary properties    
        % Acceleration due to gravity in m/s^2
        Gravity = 9.8
        m = 1
        b = 0.5
        L = 0.5
        
        % Max input
        max_input_value = 1
               
        % Sample time
        Ts = 0.02

        % Store time
        ElapsedTime = 0

        % Angle at which to fail the episode (radians)
        % AngleThreshold = 12 * pi/180
        % 
        % % Distance at which to fail the episode
        % DisplacementThreshold = 2.4
        max_time_limit = 30;

    end
    
    properties
        % Initialize system state [x1,x2]'
        State = zeros(2,1)
        ThetaHistory = [];  % History of theta values
        TimeHistory = [];
        TorqueHistory = [];% Corresponding time stamps
        PlotAxes = [];
    end
    % properties (Access = private)
    %     PlotAxes % Handle to the plot axes
    % 
    % end
    
    properties(Access = protected)
        % Initialize internal flag to indicate episode termination
        IsDone = false        
    end
    properties
    % Other existing properties...
    EpisodeCount = 0; % Tracks the number of episodes
    PlotFrequency = 100; % Specifies how often to plot (every 200 episodes)
    end

    properties
    LastAction = 0; % Initialize with zero or an appropriate default value
    end



    %% Necessary Methods
    methods              
        % Contructor method creates an instance of the environment
        % Change class name and constructor name accordingly
        function this = MyEnvironment()
            % Initialize Observation settings
            obsInfo = rlNumericSpec([2 1]);
            obsInfo.Name = 'state';  % Name of observation
            
            % Initialize Action settings   
            actionValues = -20:1:20;  % Example set of discrete actions
            actInfo = rlFiniteSetSpec(actionValues);
            actInfo.Name = 'action';
            
            % The following line implements built-in functions of RL env
            this = this@rl.env.MATLABEnvironment(obsInfo,actInfo);
            
            % Initialize property values and pre-compute necessary values
            % updateActionInfo(this);
        end
        
        % Apply system dynamics and simulates the environment with the 
        % given action for one step.
        function [Observation,Reward,IsDone,Info] = step(this,Action)
            Info = [];
            
            % Get action
            input_value = getForce(this,Action);            
            this.LastAction = Action;
            % % Unpack state vector
            % XDot = this.State(2);
            % Theta = this.State(3);
            % ThetaDot = this.State(4);
            x1 = this.State(1);
            x2 = this.State(2);
            
            % theta = x1;
            % % Cache to avoid recomputation
            % CosTheta = cos(theta);
            % SinTheta = sin(theta);            
            % SystemMass = this.CartMass + this.PoleMass;
            % temp = (Force + this.PoleMass*this.HalfPoleLength * ThetaDot^2 * SinTheta) / SystemMass;

            % % Apply motion equations            
            % ThetaDotDot = (this.Gravity * SinTheta - CosTheta* temp) / (this.HalfPoleLength * (4.0/3.0 - this.PoleMass * CosTheta * CosTheta / SystemMass));
            % XDotDot  = temp - this.PoleMass*this.HalfPoleLength * ThetaDotDot * CosTheta / SystemMass;
            

            x1_dot = x2;
            x2_dot = - (this.Gravity * sin(x1) / this.L) - (this.b * x2) / (this.m * this.L * this.L) + (input_value) / (this.m*this.L*this.L);
            % Euler integration
            Observation = this.State + this.Ts.*[x1_dot;x2_dot];
            this.ElapsedTime = this.ElapsedTime + this.Ts;
            
            % Update system states
            
            
            % Check terminal condition
            % Observation(1) = mod(Observation(1), 2*pi);
            if Observation(1) > pi
                Observation(1) = Observation(1) - 2*pi;
            elseif Observation(1) < -pi
                Observation(1) = Observation(1) + 2*pi;
            end
               

            x1 = Observation(1);
            x2 = Observation(2);
            
            this.State = Observation;
            this.ThetaHistory = [this.ThetaHistory; this.State(1)];
            this.TimeHistory = [this.TimeHistory; this.ElapsedTime];
            this.TorqueHistory = [this.TorqueHistory; this.LastAction];

            % Define the thresholds for balance
            % angleThreshold = 0.2;            % Radians around 0
            % angularVelocityThreshold = 0.1;  % Threshold for angular velocity
        
            % Check if the pendulum is balanced
            
            % IsBalanced = abs(pi - x1) < angleThreshold && abs(x2) < angularVelocityThreshold;
        
            % Check for episode termination based on time limit
            % time_limit_reached = this.ElapsedTime >= this.max_time_limit;
        
            % Calculate reward
            % if time_limit_reached
            %     if IsBalanced
            %         Reward = this.RewardForNotFalling;  % Reward for balancing within the time limit
            %     else
            %         Reward = this.PenaltyForFalling;   % Penalty for not balancing within the time limit
            %     end
            % else
            %     Reward = 0;  % No reward or penalty until the end of the time limit
            % end
            angle_error = pi - abs(x1);
            w = (abs(x2));

            % if angle_error > 30 * pi / 180
            %     Reward = - (1*angle_error);
            % else
            %     Reward = - (1 * angle_error + 2 * abs(this.State(2)) + 0.5 * Action^2);
            % end
            alpha = - 1.0; % Weight for angle error
            beta = - 0.1; % Weight for angular velocity
            gamma =  -0.001;
            Reward = alpha * angle_error^2 + beta * w ^2 + gamma * (this.LastAction)^2;
            % Reward = -cos(x1);
            % if angle_error < pi/2
            %     Reward = Reward - 10*omega_error^2;
            % end

            % (optional) use notifyEnvUpdated to signal that the 
            % environment has been updated (e.g. to update visualization)
            % this.plot();
            if mod(this.EpisodeCount, this.PlotFrequency) == 0
                this.plot(); % Plot the current state of the environment
            end

        % Optional: use notifyEnvUpdated to signal that the environment has been updated
        % notifyEnvUpdated(this);
            IsDone = this.IsDone;
            notifyEnvUpdated(this);
        end
        
        % Reset environment to initial state and output initial observation
        function InitialObservation = reset(this)
            % Random initial conditions within small range for angle (theta)
            this.EpisodeCount = this.EpisodeCount + 1;
            this.ThetaHistory = [];
            this.TimeHistory = [];
            this.TorqueHistory = [];
            
            T0 = 0.5;%(rand - 0.5)  * 10 * pi / 180;  % random value from -0.5 to 0.5
       
            
            Td0 = 0;  % Initial angular velocity
        
            InitialObservation = [T0; Td0];
            this.State = InitialObservation;
        
            % Reset the time counter
            this.ElapsedTime = 0;
        
            % Optionally update visualization
            notifyEnvUpdated(this);
            
            return
        end

    end
    %% Optional Methods (set methods' attributes accordingly)
    methods               
        % Helper methods to create the environment
        % Discrete force 1 or 2
        function force = getForce(this,action)
            if ~ismember(action,this.ActionInfo.Elements)
                error('Action must be %g for going left and %g for going right.',-this.MaxForce,this.MaxForce);
            end
            force = action;           
        end
        % update the action info based on max force
        % function updateActionInfo(this)
        %     this.ActionInfo.Elements = this.max_input_value*[-1 1];
        % end
        
        % (optional) Visualization method
        function plot(this)
            % Define the pendulum's hinge and endpoint coordinates
    % Define the coordinates of the pendulum's hinge
% Ensure the plot figure and subplots are set up
            if isempty(this.PlotAxes) || any(~isgraphics(this.PlotAxes))
                figure;
                this.PlotAxes(1) = subplot(3,1,1); % For the pendulum animation
                this.PlotAxes(2) = subplot(3,1,2); % For theta vs time plot
                this.PlotAxes(3) = subplot(3,1,3); % For torque vs time plot
            end
        
            % Clear existing plot elements
            cla(this.PlotAxes(1));
            cla(this.PlotAxes(2));
            cla(this.PlotAxes(3));
            
            % Plotting the pendulum in the first subplot
            originX = 0; originY = 0;
            theta = this.State(1);
            xEnd = originX + this.L * sin(theta);
            yEnd = originY - this.L * cos(theta);
            plot(this.PlotAxes(1), [originX, xEnd], [originY, yEnd], 'k-', 'LineWidth', 2);
            scatter(this.PlotAxes(1), xEnd, yEnd, 50, 'filled', 'r');
            scatter(this.PlotAxes(1), originX, originY, 100, 'filled', 'k');
            
            % Setting titles and labels for pendulum animation
            title(this.PlotAxes(1), 'Pendulum Simulation');
            xlabel(this.PlotAxes(1), 'X');
            ylabel(this.PlotAxes(1), 'Y');
            hold(this.PlotAxes(1), 'on');
            axis(this.PlotAxes(1), 'equal');
            xlim(this.PlotAxes(1), [-1.5 * this.L, 1.5 * this.L]);
            ylim(this.PlotAxes(1), [-1.5 * this.L, 1.5 * this.L]);
        
            % Plotting theta vs. time in the second subplot
            plot(this.PlotAxes(3), this.TimeHistory, this.ThetaHistory, 'r-');
            title(this.PlotAxes(3), 'Theta vs. Time');
            xlabel(this.PlotAxes(3), 'Time (s)');
            ylabel(this.PlotAxes(3), 'Theta (rad)');
            xlim(this.PlotAxes(3), [0, max(this.TimeHistory)+1]); % Adjust x-axis to show all data
            grid(this.PlotAxes(3), 'on');
            
            % Plotting torque vs. time in the third subplot
            
            plot(this.PlotAxes(2), this.TimeHistory, this.TorqueHistory, 'b-');
            title(this.PlotAxes(2), 'Torque vs. Time');
            xlabel(this.PlotAxes(2), 'Time (s)');
            ylabel(this.PlotAxes(2), 'Torque');
            xlim(this.PlotAxes(2), [0, max(this.TimeHistory)+1]); % Adjust x-axis to show all data
            grid(this.PlotAxes(2), 'on');
            % % Display the last action as text in the plot
            % str = sprintf('Last Action: %.2f', this.LastAction);
            % text(this.PlotAxes(1), 0.7 * max(xlim(this.PlotAxes(1))), 0.9 * max(ylim(this.PlotAxes(1))), str, ...
            %      'FontSize', 12, 'FontWeight', 'bold', 'Color', 'magenta', 'HorizontalAlignment', 'right');
        
            drawnow; % Update the plot
        
            % Optional: Notify environment update (if method is implemented)
            envUpdatedCallback(this);
        end
        
        % (optional) Properties validation through set methods
        function set.State(this,state)
            validateattributes(state,{'numeric'},{'finite','real','vector','numel',2},'','State');
            this.State = double(state(:));
            notifyEnvUpdated(this);
        end
        % function set.HalfPoleLength(this,val)
        %     validateattributes(val,{'numeric'},{'finite','real','positive','scalar'},'','HalfPoleLength');
        %     this.HalfPoleLength = val;
        %     notifyEnvUpdated(this);
        % end
        function set.Gravity(this,val)
            validateattributes(val,{'numeric'},{'finite','real','positive','scalar'},'','Gravity');
            this.Gravity = val;
        end
        function set.L(this,val)
            validateattributes(val,{'numeric'},{'finite','real','positive','scalar'},'','L');
            this.L = val;
        end
        function set.b(this,val)
            validateattributes(val,{'numeric'},{'finite','real','positive','scalar'},'','b');
            this.b = val;
        end
        function set.m(this,val)
            validateattributes(val,{'numeric'},{'finite','real','positive','scalar'},'','m');
            this.m = val;
            updateActionInfo(this);
        end
        function set.Ts(this,val)
            validateattributes(val,{'numeric'},{'finite','real','positive','scalar'},'','Ts');
            this.Ts = val;
        end
        function set.max_time_limit(this,val)
            validateattributes(val,{'numeric'},{'finite','real','positive','scalar'},'','max_time_limit');
            this.AngleThreshold = val;
        end
        % function set.DisplacementThreshold(this,val)
        %     validateattributes(val,{'numeric'},{'finite','real','positive','scalar'},'','DisplacementThreshold');
        %     this.DisplacementThreshold = val;
        % end
        % function set.RewardForNotFalling(this,val)
        %     validateattributes(val,{'numeric'},{'real','finite','scalar'},'','RewardForNotFalling');
        %     this.RewardForNotFalling = val;
        % end
        % function set.PenaltyForFalling(this,val)
        %     validateattributes(val,{'numeric'},{'real','finite','scalar'},'','PenaltyForFalling');
        %     this.PenaltyForFalling = val;
        % end
    end
    
    methods (Access = protected)
        % (optional) update visualization everytime the environment is updated 
        % (notifyEnvUpdated is called)
        function envUpdatedCallback(this)
        end
    end
end
