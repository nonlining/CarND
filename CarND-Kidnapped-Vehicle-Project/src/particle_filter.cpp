/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    num_particles = 50;

    default_random_engine gen;
    normal_distribution<double> Nd_x(x, std[0]);
    normal_distribution<double> Nd_y(y, std[1]);
    normal_distribution<double> Nd_theta(theta, std[2]);

    for(int i=0; i < num_particles; i++){
        Particle particle;
        particle.id = i;
        particle.x = Nd_x(gen);
        particle.y = Nd_y(gen);
        particle.theta = Nd_theta(gen);
        particle.weight = 1.0;
        particles.push_back(particle);
        weights.push_back(1);
    }
    is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	
  default_random_engine gen;
	
  for (int i = 0; i < particles.size(); ++i){
    double x, y, theta;
	if(yaw_rate == 0){
	  x = particles[i].x + velocity * delta_t * cos(particles[i].theta);
      y = particles[i].y + velocity * delta_t * sin(particles[i].theta);
      theta = particles[i].theta;
	} else {
	  x = particles[i].x + velocity/yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
      y = particles[i].y + velocity/yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
      theta = particles[i].theta + yaw_rate * delta_t;
	}
	normal_distribution<double> Nd_x(x, std_pos[0]);
	normal_distribution<double> Nd_y(y, std_pos[1]);
	normal_distribution<double> Nd_theta(theta, std_pos[2]);
	  
	particles[i].x = Nd_x(gen);
	particles[i].y = Nd_y(gen);
	particles[i].theta = Nd_theta(gen);
  }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	
	
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	
	double var_x = pow(std_landmark[0], 2);
	double var_y = pow(std_landmark[1], 2);
	double covar_xy = std_landmark[0] * std_landmark[1];
	double weights_sum = 0;	
	
	for (int i=0; i < num_particles; i++) {
		
		Particle& particle = particles[i];
		long double weight = 1;
		
		for (int j=0; j < observations.size(); j++) {
			
			LandmarkObs obs = observations[j];
			
			
			double predicted_x = obs.x * cos(particle.theta) - obs.y * sin(particle.theta) + particle.x;
			double predicted_y = obs.x * sin(particle.theta) + obs.y * cos(particle.theta) + particle.y;

			
			Map::single_landmark_s nearest_landmark;
			double min_distance = sensor_range;
			double distance = 0;

			// associate sensor measurements to map landmarks 
			for (int k = 0; k < map_landmarks.landmark_list.size(); k++) {

				Map::single_landmark_s landmark = map_landmarks.landmark_list[k];

				distance = fabs(predicted_x - landmark.x_f) + fabs(predicted_y - landmark.y_f);

				if (distance < min_distance) {
					min_distance = distance;
					nearest_landmark = landmark;
				}


			} // end associate nearest landmark

			double x_diff = predicted_x - nearest_landmark.x_f;
			double y_diff = predicted_y - nearest_landmark.y_f;
			double num = exp(-0.5*((x_diff * x_diff)/var_x + (y_diff * y_diff)/var_y));
			double denom = 2*M_PI*covar_xy;
			// multiply particle weight by this obs-weight pair stat
			weight *= num/denom;

		} // end observations loop

		
		particle.weight = weight;
		
		weights[i] = weight;
		
		weights_sum += weight;

}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  default_random_engine gen;
  vector<Particle> new_particles;
  discrete_distribution<int> dist_pmf(weights.begin(), weights.end());
  
  for(int i=0; i < num_particles; i++){
    new_particles.push_back(particles[dist_pmf(gen)]);
  }
  
  particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates


    particle.associations.clear();
    particle.sense_x.clear();
    particle.sense_y.clear();

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
