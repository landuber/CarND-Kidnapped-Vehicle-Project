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
    default_random_engine gen;
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);
    num_particles = 101;

    for(int i = 0; i < num_particles; i++)
    {
        Particle p;
        p.id = i;
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
        p.weight = 1.0;
        weights.push_back(1.0);
        particles.push_back(p);
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    default_random_engine gen;
    for(Particle& p : particles)
    {
        double x, y, theta;
        if(fabs(yaw_rate) > 0.001) 
        {
            x = p.x + (velocity / yaw_rate) * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta));
            y = p.y + (velocity / yaw_rate) * (cos(p.theta) - cos(p.theta + yaw_rate * delta_t));
            theta = p.theta + yaw_rate * delta_t;
        }
        else 
        {
            x = p.x + velocity * delta_t * cos(p.theta);
            y = p.y + velocity * delta_t * sin(p.theta);
            theta = p.theta;
        }

        normal_distribution<double> dist_x(x, std_pos[0]);
        normal_distribution<double> dist_y(y, std_pos[1]);
        normal_distribution<double> dist_theta(theta, std_pos[2]);
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs>& predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
    for(LandmarkObs& obs : observations)
    {
        double dist = std::numeric_limits<double>::max();
        for(LandmarkObs const& prd : predicted)
        {
            double d =  (prd.x - obs.x) * (prd.x - obs.x) +  (prd.y - obs.y) * (prd.y - obs.y);
            if(d < dist)
            {
                dist = d;
                obs.id = prd.id;
            }
        }
    }

}

double ParticleFilter::computeWeight(std::vector<LandmarkObs>& predicted, std::vector<LandmarkObs>& observations, double std_landmark[]) {

    double weight = 1.0;
    for(LandmarkObs const& obs : observations)
    {
        for(LandmarkObs const& prd : predicted)
        {
            if(prd.id == obs.id)
            {
                double exponent = -(((obs.x - prd.x) * (obs.x - prd.x)) / (2 * std_landmark[0] * std_landmark[0]) +
                                    ((obs.y - prd.y) * (obs.y - prd.y)) / (2 * std_landmark[1] * std_landmark[1]));
                weight *= (exp(exponent) / (2 * M_PI * std_landmark[0] * std_landmark[1]));
            }
        }
    }

    return weight;
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

    std::vector<LandmarkObs> predicted;
    for(auto const& lm : map_landmarks.landmark_list)
    {
        LandmarkObs lm_p;
        lm_p.id = lm.id_i;
        lm_p.x = lm.x_f;
        lm_p.y = lm.y_f;
        predicted.push_back(lm_p);
    }
    // 1, Transform landmark observations to the global reference frame
    for(Particle& p : particles)
    {
        std::vector<LandmarkObs> global_obs;
        for(LandmarkObs const& obs : observations)
        {
            if(sqrt((obs.x * obs.x) + (obs.y * obs.y)) <= sensor_range)
            {
                LandmarkObs g;
                g.x = p.x + (cos(p.theta) * obs.x) - (sin(p.theta) * obs.y);
                g.y = p.y + (sin(p.theta) * obs.x) + (cos(p.theta) * obs.y);
                global_obs.push_back(g);
            }
        }
        dataAssociation(predicted , global_obs);
        p.weight = computeWeight(predicted, global_obs, std_landmark);
        weights[p.id] = p.weight;
        //std::cout << "computed wieght " << p.weight << std::endl;
	p.associations.clear();
	p.sense_x.clear();
	p.sense_y.clear();
        for(LandmarkObs obs : global_obs)
        {
            p.associations.push_back(obs.id);
            p.sense_x.push_back(obs.x);
            p.sense_y.push_back(obs.y);
        }

    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    default_random_engine gen;
    vector<Particle> new_particles;

    // get all of the current weights
    vector<double> weights;
    for (int i = 0; i < num_particles; i++) {
    weights.push_back(particles[i].weight);
    }

    // generate random starting index for resampling wheel
    uniform_int_distribution<int> uniintdist(0, num_particles-1);
    auto index = uniintdist(gen);

    // get max weight
    double max_weight = *max_element(weights.begin(), weights.end());

    // uniform random distribution [0.0, max_weight)
    uniform_real_distribution<double> unirealdist(0.0, max_weight);

    double beta = 0.0;

    // spin the resample wheel!
    for (int i = 0; i < num_particles; i++) {
    beta += unirealdist(gen) * 2.0;
    while (beta > weights[index]) {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    new_particles.push_back(particles[index]);
    }

    particles = new_particles;
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
