/*
 * This file is part of the statismo library.
 *
 * Author: Marcel Luethi (marcel.luethi@unibas.ch)
 *
 * Copyright (c) 2011 University of Basel
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 *
 * Neither the name of the project's author nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
 * TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */


#ifndef __COME_SSM_EXCEPTIONS__
#define __COME_SSM_EXCEPTIONS__

#include <exception>
#include <string>

namespace statismo {

/**
 * \brief Used to indicate that a method has not yet been implemented
 */
class NotImplementedException : public std::exception {
  public:
    NotImplementedException(const char* classname, const char* methodname)
        :m_classname(classname), m_methodname(methodname) {
    }
    virtual ~NotImplementedException() throw() {}

    const char* what() const throw() {
        return (m_classname + "::" +m_methodname).c_str();
    }
  private:
    std::string m_classname;
    std::string m_methodname;
};

/**
 * \brief Generic Exception class for the statismo Library.
 */
class StatisticalModelException : public std::exception {
  public:
    StatisticalModelException(const char* message) : m_message(message) {}
    virtual ~StatisticalModelException() throw() {}
    const char* what() const throw() {
        return m_message.c_str();
    }

  private:
    std::string m_message;
};

}

#endif
