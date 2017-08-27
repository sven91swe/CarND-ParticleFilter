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
  if (!is_initialized) {
    num_particles = 100;

    //Creates normal (Gaussian) distributions.
    double std_x, std_y, std_theta; // Standard deviations for x, y, and psi
    std_x = std[0];
    std_y = std[1];
    std_theta = std[2];
    normal_distribution<double> dist_x(x, std_x);
    normal_distribution<double> dist_y(y, std_y);
    normal_distribution<double> dist_theta(theta, std_theta);

    //Initializes all particles
    double x_sample, y_sample, theta_sample;
    for (int i = 0; i < num_particles; i++) {
      Particle p;
      p.id = i;
      p.x = dist_x(gen);
      p.y = dist_y(gen);
      p.theta = dist_theta(gen);
      p.weight = 1.0;
      particles.push_back(p);

      weights.push_back(1.0);
    }

    is_initialized = true;
  }
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	double std_x, std_y, std_theta; // Standard deviations for x, y, and psi
	std_x = std_pos[0];
	std_y = std_pos[1];
	std_theta = std_pos[2];
	normal_distribution<double> dist_x(0, std_x);
	normal_distribution<double> dist_y(0, std_y);
	normal_distribution<double> dist_theta(0, std_theta);

	for(int i = 0; i<num_particles; i++) {
		double x, y, theta, x_new, y_new, theta_new;
		x = particles[i].x;
		y = particles[i].y;
		theta = particles[i].theta;

		//Avoid division by zero
		if(fabs(yaw_rate) < 0.00001){
			x_new = x + velocity * delta_t * cos(theta);
			y_new = y + velocity * delta_t * sin(theta);
			theta_new = theta;
		}else{
			x_new = x + velocity / yaw_rate * (sin(theta + yaw_rate * delta_t) - sin(theta));
			y_new = y + velocity / yaw_rate * (cos(theta) - cos(theta + yaw_rate * delta_t));
			theta_new = theta + yaw_rate * delta_t;
		}

		particles[i].x = x_new + dist_x(gen);
		particles[i].y = y_new + dist_y(gen);
		particles[i].theta = theta_new + dist_theta(gen);

    //cout << "X: " << x_new << " - Y: " << y_new << " - Theta: " << theta_new << endl;
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {

  /*
   * This has been implemented directly into updateWeights and combined with calculating the
   */

  // Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {

/* Used for debugging.
  for(int i=0; i<observations.size(); i++){
    double x_particle = particles[0].x;
    double y_particle = particles[0].y;
    double theta_particle = particles[0].theta;
    double x_map = observations[i].x * cos(-theta_particle) - observations[i].y * sin(-theta_particle) + x_particle;
    double y_map = observations[i].x * sin(-theta_particle) + observations[i].y * cos(-theta_particle) + y_particle;
    cout << "Observation - id: " << observations[i].id << " - X: " << x_map << " - Y: " << y_map << endl;
  }
*/
  int totalNumberOfLandmarks = map_landmarks.landmark_list.size();

  //looping over all particles
  for(int i=0; i<num_particles; i++){
    Particle p = particles[i];
    double x, y, theta;
    x = p.x;
    y = p.y;
    theta = p.theta;
    weights[i] = 1.0;
    std::vector<LandmarkObs> landMarksInRange = vector<LandmarkObs>();

    //looping over all landmarks
    for(int j=0; j<totalNumberOfLandmarks; j++) {
      double x_land_map, y_land_map;
      x_land_map = map_landmarks.landmark_list[j].x_f;
      y_land_map = map_landmarks.landmark_list[j].y_f;

      //add all landmarks in sensor range to a vector
      if (dist(x, y, x_land_map, y_land_map) < sensor_range) {
        /* Used for debugging
         * cout << "Landmark - id: " << map_landmarks.landmark_list[j].id_i << " - X: " << x_land_map << " - Y: "
             << y_land_map << endl; */
        landMarksInRange.push_back(LandmarkObs{map_landmarks.landmark_list[j].id_i, x_land_map, y_land_map});
      }
    }

    if(landMarksInRange.size() > 0) {
      for (int j = 0; j < observations.size(); j++) {
        //Standard transformation, observation from vehicle to map coordinate systems.
        double x_obs_map = observations[j].x * cos(theta) - observations[j].y * sin(theta) + x;
        double y_obs_map = observations[j].x * sin(theta) + observations[j].y * cos(theta) + y;


        //Finding the nearset landmark in sensor range from the map to this observation.
        LandmarkObs nearest = landMarksInRange[0];
        double closestDistance = dist(x_obs_map, y_obs_map, landMarksInRange[0].x, landMarksInRange[0].y);
        for (int k = 1; k < landMarksInRange.size(); k++) {
          double distance = dist(x_obs_map, y_obs_map, landMarksInRange[k].x, landMarksInRange[k].y);
          if (distance < closestDistance) {
            closestDistance = distance;
            nearest = landMarksInRange[k];
          }
        }

        double x_diff = x_obs_map - nearest.x;
        double y_diff = y_obs_map - nearest.y;
        double x_std = std_landmark[0];
        double y_std = std_landmark[1];
        double pi = 3.141592;
        weights[i] *=
                exp(-1.0 / 2.0 * (1.0 / pow(x_std, 2) * pow(x_diff, 2) + 1.0 / pow(y_std, 2) * pow(y_diff, 2))) /(2.0 * pi * x_std * y_std);
      }
    }else{
      //If particle has no landmarks in sensor range according to the map.
      weights[i] = 0;
    }

    particles[i].weight = weights[i];
  }

	// Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
}

void ParticleFilter::resample() {
  /*
   * Uses the selection wheel method.
   */


  double maxW = 0;
  for(int i=0; i<num_particles; i++){
    if(maxW < weights[i]){
      maxW = weights[i];
    }
  }

  uniform_int_distribution<int> startValueGen(0, num_particles);
  uniform_real_distribution<> betaDist(0.0, 2*maxW);

  int index = startValueGen(gen)%num_particles;

  std::vector<Particle> newParticles = vector<Particle>(num_particles);
  double beta = 0;
  for(int i=0; i<num_particles; i++){
     beta += betaDist(gen);

    while(beta > weights[index]){
      beta -= weights[index];
      index += 1;
      index %= num_particles;
    }
    newParticles[i] = {index, particles[index].x, particles[index].y, particles[index].theta, weights[index], vector<int>(), vector<double>(), vector<double>()};
  }

  particles = std::move(newParticles);
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
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
