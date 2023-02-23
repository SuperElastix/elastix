/*=========================================================================
 *
 *  Copyright UMC Utrecht and contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

// Its own header
#include "elxlog.h"

// Rename the spdlog namespace, to avoid naming conflicts.
#define spdlog elx_spdlog

// The logging library from https://github.com/gabime/spdlog
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_sinks.h>

#include <cassert>
#include <memory> // For make_shared.

namespace elastix
{
namespace
{

class log_data
{
public:
  auto &
  get_log_file_logger()
  {
    return m_loggers.log_file_logger;
  }

  auto &
  get_stdout_logger()
  {
    return m_loggers.stdout_logger;
  }

  auto &
  get_multi_logger()
  {
    return m_loggers.multi_logger;
  }

  void
  reset()
  {
    m_loggers = {};
  }

private:
  struct loggers
  {
    spdlog::logger log_file_logger{ "elastix file logger" };
    spdlog::logger stdout_logger{ "elastix stdout logger" };
    spdlog::logger multi_logger{ "elastix multi logger" };
  };

  loggers m_loggers;
};


auto &
get_log_data()
{
  static log_data static_log_data{};

  return static_log_data;
}

std::string
get_string_from_stream(const std::ostream & stream)
{
  const auto * const outputstringstream = dynamic_cast<const std::ostringstream *>(&stream);

  if (outputstringstream)
  {
    return outputstringstream->str();
  }
  assert(!"Failed to convert std::ostream pointer to std::ostringstream pointer!");
  log::error("Failed to convert std::ostream pointer to std::ostringstream pointer");
  return {};
}


spdlog::level::level_enum
to_spdlog_level_enum(const log::level log_level)
{
  switch (log_level)
  {
    case log::level::info:
    {
      return spdlog::level::info;
    }
    case log::level::warn:
    {
      return spdlog::level::warn;
    }
    case log::level::err:
    {
      return spdlog::level::err;
    }
    case log::level::off:
    {
      return spdlog::level::off;
    }
    default:
    {
      assert(!"Unsupported log level!");
      return spdlog::level::level_enum{};
    }
  }
}


void
setup_implementation(const std::string & log_filename,
                     const bool          do_log_to_file,
                     const bool          do_log_to_stdout,
                     const log::level    log_level)
{
  std::vector<spdlog::sink_ptr> log_file_sink;
  std::vector<spdlog::sink_ptr> stdout_sink;
  std::vector<spdlog::sink_ptr> multi_logger_sinks;

  if (do_log_to_file && (log_level != log::level::off))
  {
    // Throws an exception when the log file cannot be opened.
    auto sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(log_filename, true);
    sink->set_pattern("%v");

    log_file_sink = { sink };
    multi_logger_sinks.push_back(sink);
  }

  if (do_log_to_stdout)
  {
    auto sink = std::make_shared<spdlog::sinks::stdout_sink_mt>();
    sink->set_pattern("%v");

    stdout_sink = { sink };

    if (log_level != log::level::off)
    {
      multi_logger_sinks.push_back(sink);
    }
  }

  auto & data = get_log_data();

  // From here, the code is exception free, so all sinks will be set properly.
  const auto spdlog_level = to_spdlog_level_enum(log_level);

  auto & log_file_logger = data.get_log_file_logger();

  log_file_logger.sinks() = std::move(log_file_sink);
  log_file_logger.set_level(spdlog_level);

  data.get_stdout_logger().sinks() = std::move(stdout_sink);

  auto & multi_logger = data.get_multi_logger();

  multi_logger.set_level(spdlog_level);
  multi_logger.sinks() = std::move(multi_logger_sinks);
}

template <spdlog::level::level_enum spdlog_level>
void
log_to_multi_logger(const std::string & message)
{
  auto & logger = get_log_data().get_multi_logger();

  if (!logger.sinks().empty())
  {
    logger.log(spdlog_level, message);
  }
}

template <spdlog::level::level_enum spdlog_level>
void
log_to_multi_logger(const std::ostream & stream)
{
  auto & logger = get_log_data().get_multi_logger();

  if (!logger.sinks().empty())
  {
    logger.log(spdlog_level, get_string_from_stream(stream));
  }
}

} // namespace


bool
log::setup(const std::string & log_filename,
           const bool          do_log_to_file,
           const bool          do_log_to_stdout,
           const log::level    log_level)
{
  try
  {
    setup_implementation(log_filename, do_log_to_file, do_log_to_stdout, log_level);
    return true;
  }
  catch (const std::exception &)
  {
    return false;
  }
}


log::guard::guard() = default;

log::guard::guard(const std::string & log_filename,
                  const bool          do_log_to_file,
                  const bool          do_log_to_stdout,
                  const log::level    log_level)
{
  setup_implementation(log_filename, do_log_to_file, do_log_to_stdout, log_level);
}

log::guard::~guard()
{
  get_log_data().reset();
}


void
log::info(const std::string & message)
{
  log_to_multi_logger<spdlog::level::info>(message);
}

void
log::warn(const std::string & message)
{
  log_to_multi_logger<spdlog::level::warn>(message);
}

void
log::error(const std::string & message)
{
  log_to_multi_logger<spdlog::level::err>(message);
}


void
log::info(const std::ostream & stream)
{
  log_to_multi_logger<spdlog::level::info>(stream);
}

void
log::warn(const std::ostream & stream)
{
  log_to_multi_logger<spdlog::level::warn>(stream);
}

void
log::error(const std::ostream & stream)
{
  log_to_multi_logger<spdlog::level::err>(stream);
}

void
log::info_to_log_file(const std::string & message)
{
  auto & logger = get_log_data().get_log_file_logger();

  if (!logger.sinks().empty())
  {
    logger.info(message);
  }
}

void
log::info_to_log_file(const std::ostream & stream)
{
  auto & logger = get_log_data().get_log_file_logger();

  if (!logger.sinks().empty())
  {
    logger.info(get_string_from_stream(stream));
  }
}

void
log::to_stdout(const std::string & message)
{
  get_log_data().get_stdout_logger().info(message);
}

void
log::to_stdout(const std::ostream & stream)
{
  log::to_stdout(get_string_from_stream(stream));
}

} // end namespace elastix
